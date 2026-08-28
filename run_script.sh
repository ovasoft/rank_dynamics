for mode in static dynamic flat; do
    python probe_dynamics.py --config configs/cifar10.yaml \
        --run_dir outputs/cifar10/LR_r8_dsn_${mode} \
        --ref_run_dir outputs/cifar10/FR
done