''' Network context namespace. 

    Gives access to original message context for connected functions.
'''

# The node id of the sender of the current message
sender_id = None

# The topic of the current message
topic = None

# The raw message
msg = None

# For shared state messages: the action type
action = None

# For shared state messages: the action content
action_content = None
