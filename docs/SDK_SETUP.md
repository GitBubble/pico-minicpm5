# External SDK setup

[中文](SDK_SETUP.zh-CN.md)

This project does not distribute ATC, DDK, libinstsim, board libraries,
container images or `libsvp_custom.so`.

The production compiler invocation needs:

- an ATC binary supporting `--framework=5`, `--npu_arch=V101`, image-list
  calibration, online OM output and custom-op registration;
- the matching runtime/linker libraries in `LD_LIBRARY_PATH`;
- a custom-op library registering `ExtendRMSNorm`;
- legal access to an Hi3403 runtime for final qualification.

Run `pico-minicpm5 doctor` to inspect visible dependencies. Use absolute CLI
paths or an environment prepared by the installed vendor SDK. Never upload SDK
trees, `.so` files, board credentials or simulator dumps to public CI caches.

For public CI use `pico-minicpm5 build --backend fake`. Passing the fake build
only proves orchestration and manifest contracts; it says nothing about PICO
numerics or board loadability.
