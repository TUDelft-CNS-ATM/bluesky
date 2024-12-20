''' Network context namespace. 

    Gives access to original message context for connected functions.
'''
from typing import Any, Optional
from bluesky.network.common import ActionType


# For shared state messages: the action type
action: ActionType = ActionType.NoAction

# For shared state messages: the action content
action_content: Optional[Any] = None

# The node id of the sender of the current message
sender_id = b''

# The topic of the current message
topic = ''

# The raw message
msg = b''
