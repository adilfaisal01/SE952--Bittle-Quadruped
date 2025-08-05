from omni.isaac.kit import SimulationApp

app= SimulationApp({
"headless": False,
"hide_ui": False,

})

# from environment import Environment
# Environment.add_training_grounds(n=1,size=12)
# Environment.add_bittles(n=1)

# while app.is_running():
#     app.update()