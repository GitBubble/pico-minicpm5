# 可复现的执行器构建

[English](EXECUTOR_BUILD.md)

`release/v0.2.0/release-manifest.json` 以哈希绑定板端执行器。本文是从本仓库
源码复现该哈希的配方——绑定应当可被验证，而不是被信任。

```text
源码    app/native/pico_persistent_acl_executor.c
二进制  sha256 cef4edb2ca71a3fd3b2f7ef9612d8090fb25fe95a19c465cd312383cf76a0374
        37,840 字节，AArch64 ELF
```

## 工具链

受认可的板端族，以 `linux/amd64` 容器运行：

```text
镜像      svp-pico-aarch64-tc:latest
编译器    aarch64-mix210-linux-gcc (HC&C V1R3C00SPC200B042_20221123) 7.3.0
前缀      /opt/linux/x86-arm/aarch64-mix210-linux/
优化      -O2
```

SDK stub 库不随本仓库分发，请从自己的 DDK 安装中按下面的路径提供。

## 命令

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/w svp-pico-aarch64-tc:latest sh -c '
  cd /w
  mkdir -p /tmp/lib
  cp <ddk>/acllib/lib64_aarch64-mix210-linux/stub/*.so* /tmp/lib/
  ln -sf libprotobuf-c.so.1 /tmp/lib/libprotobuf-c.so
  aarch64-mix210-linux-gcc -O2 \
    -I <ddk>/acllib/include/acl \
    app/native/pico_persistent_acl_executor.c \
    -L /tmp/lib -lsvp_acl -lsvp_aicpu -lprotobuf-c -lsecurec -lpthread -ldl -lm \
    -o pico_persistent_acl_executor.aarch64'
sha256sum pico_persistent_acl_executor.aarch64
```

## 唯一的坑

没有 `libprotobuf-c.so -> libprotobuf-c.so.1` 这个符号链接，链接器会回退到 stub
目录里的静态 `libprotobuf-c.a`。构建照样成功、执行器照样能跑，但 ELF 大小不同、
哈希也不同。正确的构建带有 `NEEDED libprotobuf-c.so.1`。

## v0.2.0 为什么换了绑定

`v0.1.0` 钉的是 `b58b5c27…`，那个执行器早于常驻输入 opcode。该版本的源码无法
重建它——也就是说那条 pin 描述的是一个谁都无法复现的二进制。v0.2.0 的执行器被
三次独立构建、逐字节一致，并且在写清单之前用本文的配方对着**已提交的源码**重跑
过一次。

新执行器产出了[性能板](../release/perf/README.md)里的每一个数字：它在多次
execute 之间保留 workspace 输入而不是反复重写，从而每个 decode 步省下一次完整的
workspace 写入。节省因此与上下文成正比——ctx1024 `5.5 ms`、ctx4096 `27.6 ms`、
ctx8192 `54.9 ms`，三点折算下来都落在 `3.7–4.7 GB/s` 的避免流量上。
