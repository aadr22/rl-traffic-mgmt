import argparse
import os
import sys
import random
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
import traci

# RFID Monitoring Functions
def scan_rfids(simulation_step):
    """Simulates RFID tag scanning from vehicles."""
    rfid_data = {
        f"rfid_{lane}": traci.lane.getLastStepVehicleNumber(lane)
        for lane in traci.lane.getIDList()
    }
    print(f"Step {simulation_step}: Scanned RFIDs - {rfid_data}")
    return rfid_data

def measure_rfid_frequency(rfid_data):
    """Calculates total frequency of RFID scans."""
    return sum(rfid_data.values())

def image_recognition_check(lane_id):
    """Simulates image recognition for vehicle counting."""
    return traci.lane.getLastStepVehicleNumber(lane_id)

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Q-Learning with RFID Monitoring"
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()

    experiment_time = str(datetime.now()).split(".")[0].replace(":", "-")
    out_csv = f"outputs/2way-single-intersection/{experiment_time}_qlearning_rfid"

    os.makedirs("outputs/2way-single-intersection", exist_ok=True)

    # RFID monitoring parameters
    THRESHOLD_FREQUENCY = 5  # Adjust based on your needs
    
    env = SumoEnvironment(
        net_file="C:/Users/arahm/sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="C:/Users/arahm/sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_warnings=False,
    )

    for run in range(1, args.runs + 1):
        simulation_step = 0
        initial_states = env.reset()
        
        # Initialize Q-Learning agents
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=args.alpha,
                gamma=args.gamma,
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=args.epsilon,
                    min_epsilon=args.min_epsilon,
                    decay=args.decay
                ),
            )
            for ts in env.ts_ids
        }

        done = {"__all__": False}
        
        while not done["__all__"]:
            # Step 1: Scan RFIDs
            rfid_data = scan_rfids(simulation_step)
            
            # Step 2: Measure Frequency
            total_frequency = measure_rfid_frequency(rfid_data)
            print(f"Step {simulation_step}: Total RFID Frequency - {total_frequency}")
            
            # Step 3: Check Threshold
            if total_frequency >= THRESHOLD_FREQUENCY:
                # Step 4: Image Recognition
                for ts in env.ts_ids:
                    for lane in traci.trafficlight.getControlledLanes(ts):
                        vehicle_count = image_recognition_check(lane)
                        
                        # Step 5-7: Traffic Jam Detection and Management
                        if vehicle_count > THRESHOLD_FREQUENCY:
                            print(f"Traffic jam detected on lane {lane}")
                            current_phase_duration = traci.trafficlight.getPhaseDuration(ts)
                            # Step 8: Increase Timer
                            traci.trafficlight.setPhaseDuration(ts, current_phase_duration + 10)
            
            # Q-Learning Action Selection and Learning
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
            s, r, done, _ = env.step(action=actions)
            
            # Update Q-values
            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(
                    next_state=env.encode(s[agent_id], agent_id),
                    reward=r[agent_id]
                )
            
            simulation_step += 1

        env.save_csv(out_csv, run)
        env.close()

print("Simulation completed successfully.")
