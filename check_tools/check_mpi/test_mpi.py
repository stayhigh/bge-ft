from mpi4py import MPI
print(f"[staring] Rank {comm.Get_rank()} of {comm.Get_size()}")
comm = MPI.COMM_WORLD
print(f"Rank {comm.Get_rank()} of {comm.Get_size()}")

