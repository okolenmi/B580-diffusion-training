"""Infrastructure layer -- concrete implementations of the application ports.

Depends on the application layer (to implement its ABCs) and on
third-party/OS facilities (sqlite3, threads). Nothing in here is
imported by domain or application -- the composition root wires it in.
"""
