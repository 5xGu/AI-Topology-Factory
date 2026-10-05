'''
Importing graph.py triggers every node/condition type registration as a
side effect (see agentic_dd.agentlib.graph_build's explicit import block).
All tests in this suite rely on that having happened first.
'''
import agentic_dd.agentlib.graph  # noqa: F401