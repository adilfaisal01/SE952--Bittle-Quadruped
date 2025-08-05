from omni.isaac.kit import SimulationApp

app= SimulationApp({
"headless": False,
"hide_ui": False,

})


from environment import Environment
e=Environment()
print("1",flush=True)
e.add_training_grounds(n=1,size=12)
print("2",flush=True)
e.add_bittles(n=1)
print("3",Flush=True)

while app.is_running():
    app.update()