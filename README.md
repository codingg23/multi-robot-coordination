# multi-robot-coordination

3-4 Raspberry Pi robots that work together to map unknown environments and find objects.
Each robot has a camera, ultrasonic sensors, and communicates over a local WiFi network.
Built as a personal project in 2024 during high school.

The robots use computer vision to recognise landmarks and detect obstacles, share their
observations over a shared state server, and use a simple learned policy to decide where
to explore next. The end goal was "search and retrieve" - send the swarm into a room,
have it find a specific coloured object, and signal its location.

It mostly works. The mapping part is solid. The RL part is rough - I did not have enough
compute to train properly, so the policy is more of a heuristic with learned weights.

## What it does

- **Collaborative mapping**: each robot builds a local occupancy grid and pushes updates
  to a central state server. Robots avoid re-exploring cells already covered by others.

- **Object detection**: YOLOv8-nano running on each Pi (it just barely fits). Detects
  coloured objects and broadcasts their estimated position to the swarm.

- **Obstacle avoidance**: reactive layer using ultrasonic sensors, runs on a separate
  thread so it can't be blocked by the vision or planning code.

- **Coordination policy**: trained with PPO. Observation is the local occupancy grid +
  positions of other robots + known object locations. Action is a heading + speed command.

- **Search and retrieve**: once all robots have seen the target object at least once, the
  nearest robot is assigned to go get it while others hold position.

## Hardware

Each robot:
- Raspberry Pi 4 (4GB)
- Pi Camera Module 3
- HC-SR04 ultrasonic sensors (front + two sides)
- L298N motor driver + two DC motors
- 5Ah USB-C power bank
- Custom 3D-printed chassis (files in `hardware/`)

State server runs on a laptop or a Pi 5 connected to the same network.

## Running it

```bash
# on state server
python server/state_server.py --port 5555

# on each robot (set ROBOT_ID=0,1,2,3)
ROBOT_ID=0 python main.py --server 192.168.1.100:5555 --mode explore
```

## Training the policy

Simulation environment is in `rl/sim_env.py` - runs faster-than-real-time for training.

```bash
python rl/train.py --timesteps 500000 --n-robots 3
```

I trained on a laptop (M1 MacBook) for about 6 hours. The policy generalises okay to
rooms it hasn't seen before as long as they're not too different in size.

## File layout

```
vision/          - object detection and landmark recognition
navigation/      - occupancy grid, path planning
coordination/    - state server and robot state sync
rl/              - PPO environment + training script
utils/           - sensor drivers, motor control, logging
hardware/        - chassis STL files and wiring diagrams
tests/           - unit tests for grid math and comms protocol
```

## Known issues

- The RL policy doesn't handle >4 robots well, training becomes unstable
- Camera latency on Pi 4 is ~180ms end to end, causes issues at higher speeds
- Object position estimates drift without correction (no visual odometry)
- The server is a single point of failure - if it goes down, robots freeze

## What I'd do differently

After doing this I'd probably use a proper multi agent RL framework rather than rolling
my own. Also proper SLAM instead of the simple occupancy grid would fix the drift issue.
The hardware design works but the wiring is messy - would use a custom PCB next time.

## Training results

After 500k steps with 3 robots: mean episode reward +127, coverage ~82%, target found in 78% of episodes.
