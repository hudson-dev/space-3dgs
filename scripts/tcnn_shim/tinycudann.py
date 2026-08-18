# Optional shim (see README "Troubleshooting"): put this directory on PYTHONPATH
# when the installed tinycudann has no prebuilt extension for your GPU and raises
# an EnvironmentError that nerfstudio's importer does not catch. space-3dgs is
# gsplat-based and never needs tinycudann, so we shadow it with a stub that
# raises ModuleNotFoundError -> nerfstudio sets TCNN_EXISTS=False and proceeds.
# The real installation is untouched.
raise ModuleNotFoundError("tinycudann shimmed out (not needed for space-3dgs)")
