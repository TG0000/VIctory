//! End-to-end tests of a buffer's record count read from its descriptor
//! (`buf[size - 1]` through `s_buffer_load`): a synthetic GFX10 program
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the host
//! GPU and WARP.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// Stores the record count of the buffer at table entry `entry`, and dword 1
/// of its element `count - 1` (elements of 20 bytes), to the raw UAV at entry 2.
fn program(entry: u32) -> Vec<u32> {
    vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0000 | entry << 5, // s_load_dwordx4 s[12:15], s[4:5], entry
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x8110_C10E, // s_add_i32 s16, s14, -1
        0x9310_FF10,
        20, // s_mul_i32 s16, s16, 20
        0xF420_0446,
        0x2000_0004, // s_buffer_load_dword s17, s[12:15], s16 offset:4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_020E, // v_mov_b32 v1, s14
        0x7E04_0211, // v_mov_b32 v2, s17
        0xE070_0000,
        0x800A_0100, // buffer_store_dword v1, off, s[40:43], 0
        0xE070_0004,
        0x800A_0200, // buffer_store_dword v2, off, s[40:43], 0 offset:4
        0xBF81_0000, // s_endpgm
    ]
}

/// Lowers [`program`] reading table entry `entry`, with the metadata naming
/// the buffer `kind`, `stride` bytes an element, at dword `at` of the table.
fn build(kind: ResourceType, stride: u32, at: u32, entry: u32) -> Result<HostShader, ShaderError> {
    let range = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![range(kind, at, stride), range(ResourceType::Uav, 16, 0)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: program(entry)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// Runs `shader` over `data` viewed as `stride`, on the hardware GPU and on
/// WARP (each skipped without a device), and returns what it stored.
fn run(shader: &HostShader, data: &[u32], stride: Stride) -> Vec<(&'static str, [u32; 2])> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // The table starts at heap index 0.
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]),
                    )],
                    views: &[
                        View::Buffer { data, stride },
                        // (The UAV, at table entry 2.)
                        View::Buffer { data, stride },
                        View::RawUav(2),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, [words[0][0], words[0][1]]));
    }
    out
}

/// Seven elements of 20 bytes: the count is 7 elements (not 140 bytes), and
/// the last element is found with it.
#[test]
fn a_structured_buffers_record_count_is_its_elements() {
    let shader = build(ResourceType::Srv, 20, 0, 0).expect("lower");
    let data: Vec<u32> = (0..35).map(|i| 1000 + i).collect();
    for (label, got) in run(&shader, &data, Stride::Structured(20)) {
        assert_eq!(got, [7, 1000 + 6 * 5 + 1], "{label}");
    }
}

/// A raw buffer's count is its bytes (the last "element" of the program is
/// then past its end and reads zero).
#[test]
fn a_raw_buffers_record_count_is_its_bytes() {
    let shader = build(ResourceType::Srv, 1, 0, 0).expect("lower");
    let data: Vec<u32> = (0..35).collect();
    for (label, got) in run(&shader, &data, Stride::Raw) {
        assert_eq!(got, [140, 0], "{label}");
    }
}

/// What the count counts must be known: not for constant buffers, strides no
/// host view has, or a descriptor the metadata does not name (the buffer's
/// entry starts after it, or is an earlier one).
#[test]
fn unknown_record_counts_stop_the_program() {
    for (kind, stride, at, entry) in [
        (ResourceType::Cbv, 0, 0, 0),
        (ResourceType::Srv, 0, 0, 0),
        (ResourceType::Srv, 6, 0, 0),
        (ResourceType::Srv, 4096, 0, 0),
        (ResourceType::Srv, 20, 4, 0),
        (ResourceType::Srv, 20, 0, 1),
    ] {
        let e = build(kind, stride, at, entry).err();
        assert!(
            matches!(e, Some(ShaderError::Unsupported { .. })),
            "{kind:?} {stride} {at} {entry}: {e:?}"
        );
    }
}
