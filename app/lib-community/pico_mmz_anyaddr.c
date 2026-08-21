/* SPDX-License-Identifier: Apache-2.0
 * Community ACL issues IOC_MMB_ALLOC_V3 with start=0 (below the MMZ
 * zone). Rewrite the start to a bump pointer inside the zone, then
 * remember phys/size so later SYS_FLUSH_CACHE uses the real address
 * instead of the original zero.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>

#ifndef _IOC_NRBITS
#define _IOC_NRBITS 8
#endif

#define MMZ_BASE 0x1D0000000ULL
#define TAB 64

/* userspace mmb_info without __KERNEL__ fields; 120 bytes on aarch64 */
struct mmb_info {
    uint64_t phys_addr;
    uint64_t align;
    uint64_t size;
    uint32_t order;
    uint32_t pad;
    void *mapped;
    unsigned long w32_stuf;
    char mmb_name[32];
    char mmz_name[32];
    unsigned long gfp;
} __attribute__((aligned(8)));

struct slot {
    uint64_t phys;
    uint64_t size;
    void *mapped;
};

static struct slot g_tab[TAB];
static uint64_t g_bump = MMZ_BASE;

static void remember_phys(uint64_t phys, uint64_t size)
{
    unsigned i;
    for (i = 0; i < TAB; i++) {
        if (g_tab[i].phys == 0) {
            g_tab[i].phys = phys;
            g_tab[i].size = size;
            g_tab[i].mapped = NULL;
            return;
        }
    }
}

static void remember_map(uint64_t phys, void *mapped)
{
    unsigned i;
    for (i = 0; i < TAB; i++) {
        if (g_tab[i].phys == phys) {
            g_tab[i].mapped = mapped;
            return;
        }
    }
}

static uint64_t phys_for_map(void *mapped, uint64_t fallback)
{
    unsigned i;
    uintptr_t m = (uintptr_t)mapped;
    for (i = 0; i < TAB; i++) {
        uintptr_t base = (uintptr_t)g_tab[i].mapped;
        if (g_tab[i].mapped != NULL && m >= base &&
            m < base + (uintptr_t)g_tab[i].size)
            return g_tab[i].phys + (uint64_t)(m - base);
    }
    return fallback;
}

int ioctl(int fd, unsigned long request, ...)
{
    static int (*real_ioctl)(int, unsigned long, ...);
    va_list ap;
    struct mmb_info *info;
    unsigned nr;
    int ret;

    if (real_ioctl == NULL) {
        real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    }
    va_start(ap, request);
    info = va_arg(ap, struct mmb_info *);
    va_end(ap);

    nr = request & ((1u << _IOC_NRBITS) - 1u);
    if (_IOC_TYPE(request) == 'm' && nr == 14u && info != NULL) {
        if (info->phys_addr < MMZ_BASE) {
            uint64_t need = (info->size + 0xfffULL) & ~0xfffULL;
            if (need == 0)
                need = 0x1000;
            info->phys_addr = g_bump;
            g_bump += need;
        }
    }
    if (_IOC_TYPE(request) == 'm' && nr == 24u && info != NULL) {
        uint64_t phys = phys_for_map(info->mapped, info->phys_addr);
        if (phys >= MMZ_BASE)
            info->phys_addr = phys;
    }

    ret = real_ioctl(fd, request, info);

    if (ret == 0 && _IOC_TYPE(request) == 'm' && info != NULL) {
        if (nr == 14u || nr == 10u || nr == 13u)
            remember_phys(info->phys_addr, info->size);
        if (nr == 15u || nr == 16u)
            remember_map(info->phys_addr, info->mapped);
    }
    return ret;
}
