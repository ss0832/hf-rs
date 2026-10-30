//! Manual profiling to separate integral, Fock and SCF costs.
use hf_rs::*;
use std::time::Instant;

#[test]
#[ignore="manual performance measurement"]
fn profile_cartesian_core_costs() {
    let mut atoms=Vec::new();
    for (symbol,radius) in [("C",1.39),("H",2.47)] {for k in 0..6 {
        let angle=std::f64::consts::PI*k as f64/3.0;
        atoms.push(Atom::from_angstrom(symbol,[radius*angle.cos(),radius*angle.sin(),0.0]));
    }}
    let basis=build_basis(&atoms,&get_builtin_basis("6-31g*").unwrap());
    let mut times=serde_json::Map::new();
    let start=Instant::now();
    let (s,_,_,h)=scf::compute_one_electron_integrals(&atoms,&basis);
    times.insert("one_electron".into(),serde_json::json!(start.elapsed().as_secs_f64()));
    let start=Instant::now();let shell=shell_integrals::compute_shell_eri_tensor(&basis).unwrap();
    times.insert("shell_eri".into(),serde_json::json!(start.elapsed().as_secs_f64()));
    let start=Instant::now();let pairs=scf::compute_eri_tensor_cartesian_pairs(&basis);
    times.insert("ao_pair_eri".into(),serde_json::json!(start.elapsed().as_secs_f64()));
    let worst=shell.iter().zip(&pairs).map(|(a,b)|(a-b).abs()).fold(0.0_f64,f64::max);
    assert!(worst<1e-10,"ERI difference {worst}");
    let x=s.canonical_orthogonalizer(1e-8).unwrap();
    let (_,cp)=x.transpose().dot(&h).dot(&x).eigh();let c=x.dot(&cp);
    let p=c.occupied_density(21,2.0);
    let start=Instant::now();for _ in 0..10 {std::hint::black_box(scf::build_fock_matrix(&h,&p,&shell,basis.len()));}
    times.insert("fock_average".into(),serde_json::json!(start.elapsed().as_secs_f64()/10.0));
    let start=Instant::now();let sad=sad::compute_sad_density(&atoms,&basis,0,&s).unwrap();
    times.insert("sad_cold".into(),serde_json::json!(start.elapsed().as_secs_f64()));
    assert!((sad.dot(&s).trace()-42.0).abs()<1e-9);
    let start=Instant::now();let hf=run_scf_general(&atoms,&basis,0,100,1e-10,ConvAlgo::Cdiis,None).unwrap();
    assert!(hf.converged);
    times.insert("scf_warm_sad".into(),serde_json::json!(start.elapsed().as_secs_f64()));
    let out=serde_json::json!({"system":"benzene/6-31g* Cartesian","threads":rayon::current_num_threads(),
        "n_ao":basis.len(),"timings_seconds":times,"eri_max_difference":worst,"energy":hf.total_energy});
    eprintln!("{}",serde_json::to_string_pretty(&out).unwrap());
    std::fs::write(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/core_timing.json"),serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
