"""Documented native action identifiers grouped for animation workflows."""

GROUPS = {
    "history": ("Scene.Undo", "Scene.Redo"),
    "physics": (
        "AutoPhysicsTool.Switch Auto Physics",
        "AutoPhysicsTool.Snap to Auto Physics",
        "AutoPhysicsTool.Set priority frame",
        "AutoPhysicsTool.Switch Frozen Auto Physics",
        "AutoPhysicsTool.Show all fucrum points",
        "View.FixFoot",
        "View.FixFoot_KeepKeyFrames",
        "BallisticTrajectoryTool.Add ballistic trajectory",
        "BallisticTrajectoryTool.Snap centers of mass to selected trajectory",
        "BallisticTrajectoryTool.Switch ballistic ghosts",
    ),
    "posing": (
        "AutoPosingTool.AutoPosing",
        "AutoPosingTool.Update",
        "AutoPosingTool.AutoUnlock",
        "AutoPosingTool.SwitchLock",
        "AutoPosingTool.SwitchFingersDrawing",
        "Scene.Edit mode.Current frame",
        "Scene.Edit mode.Selection",
        "Scene.Set key on change",
        "MirrorTool.Mirror on current frame",
        "MirrorTool.Mirror on interval",
    ),
    "timing": (
        "View.Animation unbaking",
        "View.AutoInterpolation_Keys",
        "View.AutoInterpolation_Interpolations",
        "View.Retargeting_Copy",
        "View.Retargeting_Paste",
        "Timeline.Play",
        "Timeline.Next key frame",
        "Timeline.Previous key frame",
        "Timeline.Create cycle",
        "Timeline.Create cycle with position and rotation offsets",
        "Timeline.Remove cycles",
        "Timeline.Change to FK key",
        "Timeline.Change to IK key",
        "Timeline.Change to fulcrum key",
        "Copier.Copy",
        "Copier.Paste",
        "Copier.Copy interval",
        "Copier.Paste interval",
    ),
    "review": (
        "View.Center camera",
        "View.Silhouette mode",
        "GhostTool.Keyframe ghosts",
        "GhostTool.Neighbor frame ghosts",
        "GhostTool.Disable ghost",
        "Window.Graph editor",
        "Window.Event log",
        "Window.Physics settings",
    ),
}

ACTIONS = {
    identifier: {"category": category, "description": identifier.split(".")[-1]}
    for category, identifiers in GROUPS.items()
    for identifier in identifiers
}
