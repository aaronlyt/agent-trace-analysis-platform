"""sandbox -- toy research QA sandbox (env + scripted policy + fault injection + targeted replay).

Depends only on atap.core (layered invariants); assembled by runtime/cli/demo,
algorithm modules access replay capabilities through RunContext.env
(ReplayEnvironment protocol).
"""

from atap.sandbox.policy import ToySandbox

__all__ = ["ToySandbox"]
