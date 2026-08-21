/* SPDX-License-Identifier: Apache-2.0
 * Symbols community libsvp_acl.so expects from the MPI sample link line.
 * libot_sys/libot_base cannot be preloaded on a live Orange Pi desktop
 * (constructors re-register modules and crash). Provide the handful of
 * lookups ACL does at load time.
 */
#include <stdint.h>
#include <stddef.h>

int osal_init(void);

__attribute__((constructor(101)))
static void pico_osal_ctor(void)
{
    (void)osal_init();
}

int32_t ot_mpi_sys_get_cur_pts(uint64_t *cur_pts)
{
    if (cur_pts)
        *cur_pts = 0;
    return 0;
}

int32_t ot_mpi_sys_flush_cache(uint64_t phys, void *virt, uint32_t size)
{
    (void)phys;
    (void)virt;
    (void)size;
    return 0;
}

int ot_sys_mod_init(void) { return 0; }
int ot_sys_mod_exit(void) { return 0; }
int ot_base_mod_init(void) { return 0; }
int ot_base_mod_exit(void) { return 0; }

int cmpi_register_module(void *modules)
{
    (void)modules;
    return 0;
}

void cmpi_unregister_module(int mod_id) { (void)mod_id; }

void *cmpi_get_module_func_by_id(int mod_id)
{
    (void)mod_id;
    return NULL;
}
