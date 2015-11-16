import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

def create_data(N):
    return host_x, x

if __name__ == "__main__":
    N = 1e7

    platforms = cl.get_platforms()
    devices = [d for platform in platforms for d in platform.get_devices()]
    for i, d in enumerate(devices):
        print("#{0}: {1} on {2}".format(i, d.name, d.platform.name))
    ctx = cl.Context(devices[2:])

    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    program = cl.Program(ctx, open('sum.cl').read()).build(options='')

    host_x = np.random.rand(N).astype(np.float32)
    x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_x)

    times = {}
    for num_workgroups in 2 ** np.arange(3, 10):
        partial_sums = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * num_workgroups)
        host_partial = np.empty(num_workgroups).astype(np.float32)
        for num_workers in 2 ** np.arange(2, 6):
            local = cl.LocalMemory(num_workers * 4)
            event = program.sum_coalesced(queue, (num_workgroups * num_workers,), (num_workers,),
                                          x, partial_sums, local, np.uint64(N))
            cl.enqueue_copy(queue, host_partial, partial_sums, is_blocking=True)

            sum_gpu = sum(host_partial)
            sum_host = sum(host_x)
            seconds = (event.profile.end - event.profile.start) / 1e9
            assert abs((sum_gpu - sum_host) / max(sum_gpu, sum_host)) < 1e-4,"GPU sum: %r, host sum: %r" % (sum_gpu, sum_host)
            times['coalesced', num_workgroups, num_workers] = seconds
            print("coalesced reads, workgroups: {}, num_workers: {}, {} seconds".
                  format(num_workgroups, num_workers, seconds))

    for num_workgroups in 2 ** np.arange(3, 10):
        partial_sums = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * num_workgroups)
        host_partial = np.empty(num_workgroups).astype(np.float32)
        for num_workers in 2 ** np.arange(2, 6):
            local = cl.LocalMemory(num_workers * 4)
            event = program.sum_blocked(queue, (num_workgroups * num_workers,), (num_workers,),
                                        x, partial_sums, local, np.uint64(N))
            cl.enqueue_copy(queue, host_partial, partial_sums, is_blocking=True)

            sum_gpu = sum(host_partial)
            sum_host = sum(host_x)
            seconds = (event.profile.end - event.profile.start) / 1e9
            assert abs((sum_gpu - sum_host) / max(sum_gpu, sum_host)) < 1e-4,"GPU sum: %r, host sum: %r" % (sum_gpu, sum_host)
            times['blocked', num_workgroups, num_workers] = seconds
            print("blocked reads, workgroups: {}, num_workers: {}, {} seconds".
                  format(num_workgroups, num_workers, seconds))

    # intensity plot of times vs. # of workgroups & workers
    matrix_coalesced = np.zeros((7, 4))
    for i in range(7):
        for k in range(4):
            matrix_coalesced[i, k] = times['coalesced', 2**(i+3), 2**(k+2)]
    matrix_blocked = np.zeros((7, 4))
    for i in range(7):
        for k in range(4):
            matrix_blocked[i, k] = times['blocked', 2**(i+3), 2**(k+2)]
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix_coalesced, cmap='RdPu', interpolation='nearest', origin='lower', extent=[2, 5.9, 3, 9.9])
    ax.set_yticks(np.arange(3,10))
    ax.set_xticks(np.arange(2,6))
    ax.set_ylabel('# of workgroups ($2^n$)')
    ax.set_xlabel('# of workers ($2^n$)')
    ax.set_title('Time to sum vs. number of workgroups & workers')
    cbar = fig.colorbar(cax)
    plt.savefig('coalesced.png')
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix_blocked, cmap='RdPu', interpolation='nearest', origin='lower', extent=[2, 5.9, 3, 9.9])
    ax.set_yticks(np.arange(3,10))
    ax.set_xticks(np.arange(2,6))
    ax.set_ylabel('# of workgroups ($2^n$)')
    ax.set_xlabel('# of workers ($2^n$)')
    ax.set_title('Time to sum vs. number of workgroups & workers')
    cbar = fig.colorbar(cax)
    plt.savefig('blocked.png')

    best_time = min(times.values())
    best_configuration = [config for config in times if times[config] == best_time]
    print("configuration {}: {} seconds".format(best_configuration[0], best_time))
