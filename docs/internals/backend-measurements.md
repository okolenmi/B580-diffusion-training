# Backend measurements

Numbers that used to sit in code comments. The code says *why* a constant or a design is what it is; the evidence for it
is here.

## Memory ledger defaults (B580, 12,216 MB)

- A desktop session holds about 950 MB of the card, and an idle device context costs about 600 MB of driver reading above
  what the allocator reports as reserved. Hence `DEFAULT_FOREIGN_RESERVE_MB = 1024` and `DEFAULT_PROCESS_OVERHEAD_MB = 600`.

## Upload streaming (assets)

- Buffering a 600 MB upload and joining its chunks stalled the event loop for 971 ms (concurrent requests saw 9 ms rise to
  971 ms) and held about twice the file in RAM. With an 8 GiB cap that is not survivable, so the write is streamed.

## Execution scratch (what a run leaves on disk)

One 4000-node execution is 845,756 bytes: the event file 523,692 (62%, 131 bytes per node), the graph 321,817 (38%), the
log 247. A monitor report is 126 bytes; one per second is about 0.5 MB per hour, about 5 MB over 12 h. The size of one run
is small; the number of runs is unbounded (a hundred runs is 85 MB, a thousand 850 MB), which is why the startup sweep
removes the scratch of finished runs.

## Signals in the graph child

- Stopped mid-run on a 60000-node graph: SIGINT exits 0 with the outcome record written (200 node records); an unhandled
  SIGTERM exits -15 with no outcome record (86 node records). Both are now handled the same way.
- Argument parsing and imports before the handlers were installed took 37 ms: small enough that nobody hit it by hand.

## Device probe

- Each of the eight device questions the installer page asks initialises the accelerator runtime once; behind the cache it
  is one probe per interval. First-run package installation costs 7 s and 32 MB on the development machine.
