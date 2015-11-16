import pyopencl as cl
import numpy as np
from matplotlib import pyplot as plt

def create_data(N):
    return host_x, x

if __name__ == "__main__":
    N = 1e7

    platforms = cl.get_platforms()
    devices = [d for platform in platforms for d in platform.get_devices()]
    for i, d in enumerate(devices):
        print("#{0}: {1} on {2}".format(i, d.name, d.platform.name))
    ctx = cl.Context([devices[2]])

    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    program = cl.Program(ctx, open('sum.cl').read()).build(options='')

    host_x = np.random.rand(N).astype(np.float32)
    x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_x)

    times = {}

    n_workgrp_list = 2 ** np.arange(3, 10)
    n_workers_list = 2 ** np.arange(2, 8)
    for num_workgroups in n_workgrp_list:
        partial_sums = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * num_workgroups)
        host_partial = np.empty(num_workgroups).astype(np.float32)
        for num_workers in n_workers_list:
            local = cl.LocalMemory(num_workers * 4)
            event = program.sum_coalesced(queue, (num_workgroups * num_workers,), (num_workers,),
                                          x, partial_sums, local, np.uint64(N))
            cl.enqueue_copy(queue, host_partial, partial_sums, is_blocking=True)

            sum_gpu = sum(host_partial)
            sum_host = sum(host_x)
            seconds = (event.profile.end - event.profile.start) / 1e9
            assert abs((sum_gpu - sum_host) / max(sum_gpu, sum_host)) < 1e-4
            times['coalesced', num_workgroups, num_workers] = seconds
            print("coalesced reads, workgroups: {}, num_workers: {}, {} seconds".
                  format(num_workgroups, num_workers, seconds))

    for num_workgroups in n_workgrp_list:
        partial_sums = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * num_workgroups)
        host_partial = np.empty(num_workgroups).astype(np.float32)
        for num_workers in n_workers_list:
            local = cl.LocalMemory(num_workers * 4)
            event = program.sum_blocked(queue, (num_workgroups * num_workers,), (num_workers,),
                                        x, partial_sums, local, np.uint64(N))
            cl.enqueue_copy(queue, host_partial, partial_sums, is_blocking=True)

            sum_gpu = sum(host_partial)
            sum_host = sum(host_x)
            seconds = (event.profile.end - event.profile.start) / 1e9
            assert abs((sum_gpu - sum_host) / max(sum_gpu, sum_host)) < 1e-4
            times['blocked', num_workgroups, num_workers] = seconds
            print("blocked reads, workgroups: {}, num_workers: {}, {} seconds".
                  format(num_workgroups, num_workers, seconds))

    best_time = min(times.values())
    best_configuration = [config for config in times if times[config] == best_time]
    print("configuration {}: {} seconds".format(best_configuration[0], best_time))

    
    # define colormap
    cmap_block = plt.cm.Blues
    cmaplist = [cmap_block(i) for i in range(cmap_block.N)]
    cmap_block = cmap_block.from_list('Custom cmap', cmaplist, 2 + len(n_workgrp_list))
    
    cmap_coa = plt.cm.Reds
    cmaplist = [cmap_coa(i) for i in range(cmap_coa.N)]
    cmap_coa = cmap_coa.from_list('Custom cmap', cmaplist, 2 + len(n_workgrp_list))
    
    
    def plot_sum_res(sum_type, cmap):
        for n_workgrp_i, n_workgrp in enumerate(n_workgrp_list):
            keys_sorted = sorted([x for x in times.keys() 
                                  if x[1] == n_workgrp 
                                  and x[0] == sum_type])
            plt.plot(n_workers_list, np.array([times[x] for x in keys_sorted]), 
                     '.-', linewidth=2,
                     color=cmap(n_workgrp_i + 2))
    plot_sum_res('coalesced', cmap_coa)
    plot_sum_res('blocked', cmap_block)
    plt.yscale('log')
    plt.xlabel('# Workers')
    plt.ylabel('Time taken (sec)');
    plt.title('Comparison of coalesced (red) vs blocked (blue) sum')
    plt.savefig('P3.png')