import argparse
import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch

from agent import DDQNAgent
from osm_traffic_handler import Traffic

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

def create_output_directory(base_output_dir, scenario_name):
    """Create output directory structure"""
    experiment_date = str(datetime.now()).split(".")[0].replace(":", "_")
    output_dir = os.path.join(base_output_dir,scenario_name,experiment_date)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def calculate_state_size(env):
    """Calculate state size from environment observation space"""
    # Get a sample observation to determine the actual state size
    initial_obs = env.reset()
    if isinstance(initial_obs, dict):
        # For multi-agent environments, get the first agent's observation
        first_agent = list(initial_obs.keys())[0]
        sample_obs = initial_obs[first_agent]
    else:
        sample_obs = initial_obs
    
    # Flatten the observation to get the actual size
    if isinstance(sample_obs, dict):
        flat_obs = np.array(list(sample_obs.values())).flatten()
    elif isinstance(sample_obs, np.ndarray):
        flat_obs = sample_obs.flatten()
    else:
        flat_obs = np.array([sample_obs]).flatten()
    
    return len(flat_obs)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Multi-Agent DDQN for OSM Traffic Signal Control"
    )
    
    # OSM and file arguments
    parser.add_argument("-osm_dir", dest="osm_dir", type=str, default="new_delhi",
                       help="Name of .net.xml directory")
    parser.add_argument("-net", dest="net_file", type=str, default="osm.net.xml",
                       help="Path to .net.xml file")
    parser.add_argument("-route", dest="route_file", type=str, default="osm.rou.xml",
                       help="Path to .rou.xml file")
    
    # DDQN parameters
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.0001,
                       help="Learning rate for DDQN")
    parser.add_argument("-gamma", dest="gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("-epsilon", dest="epsilon", type=float, default=.7,
                       help="Initial epsilon for exploration")
    parser.add_argument("-epsilon_min", dest="epsilon_min", type=float, default=0.01,
                       help="Minimum epsilon")
    parser.add_argument("-epsilon_decay", dest="epsilon_decay", type=float, default=0.95,
                       help="Epsilon decay rate")
    parser.add_argument("-buffer_size", dest="buffer_size", type=int, default=3600,
                       help="Experience replay buffer size")
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("-target_update", dest="target_update_freq", type=int, default=100,
                       help="Target network update frequency")
    
    # Traffic light parameters
    parser.add_argument("-min_green", dest="min_green", type=int, default=10,
                       help="Minimum green time")
    parser.add_argument("-max_green", dest="max_green", type=int, default=50,
                       help="Maximum green time")
    parser.add_argument("-yellow_time", dest="yellow_time", type=int, default=3,
                       help="Yellow phase duration")
    parser.add_argument("-delta_time", dest="delta_time", type=int, default=10,
                       help="Time between actions")
    
    # Simulation parameters
    parser.add_argument("-gui", action="store_true", default=False,
                       help="Run with SUMO GUI")
    parser.add_argument("-episodes", dest="episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("-max_steps", dest="max_steps", type=int, default=900,
                       help="Maximum steps per episode")
    parser.add_argument("-reward", dest="reward", type=str, default="diff-waiting-time",
                       help="Reward function")
    
    # Output and monitoring
    parser.add_argument("-output_dir", dest="output_dir", type=str, default="outputs",
                       help="Base output directory")
    parser.add_argument("-save_freq", dest="save_freq", type=int, default=100,
                       help="Model save frequency (episodes)")
    parser.add_argument("-verbose", action="store_true", default=False,
                       help="Verbose output")
    parser.add_argument("-eval_freq", dest="eval_freq", type=int, default=100,
                       help="Evaluation frequency (episodes)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
   
    # Create output directory
    output_base_dir = create_output_directory(args.output_dir, args.osm_dir)
    model_dir = os.path.join(output_base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate experiment identifier
    experiment_time = str(datetime.now()).split(".")[0].replace(":", "_")
    out_csv_name = os.path.join(
        output_base_dir,
        f"ddqn_lr{args.learning_rate}_gamma{args.gamma}_eps{args.epsilon}_decay{args.epsilon_decay}"
    )
    
    print(f"Results will be saved to: {out_csv_name}")
    print(f"Models will be saved to: {model_dir}")
    
    # Create SUMO environment
    try:     
        #path to real_traffic
        path=os.path.dirname(os.path.dirname(__file__))
        net_file=os.path.join(path,"osm_nets",args.osm_dir,args.net_file)
        route_file=os.path.join(path,"osm_nets",args.osm_dir,args.route_file)
        
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=out_csv_name,
            use_gui=args.gui,
            num_seconds=args.max_steps,
            min_green=args.min_green,
            max_green=args.max_green,
            yellow_time=args.yellow_time,
            delta_time=args.delta_time,
            reward_fn=args.reward,
            sumo_warnings=False            
        )

        if len(env.ts_ids) == 0:
            raise RuntimeError("No traffic lights found in network. Rebuild net with TLS (e.g., --tls.guess-signals/--tls.join) or pick an area with signals.")
        print(f"Environment created successfully!")
        print(f"Number of traffic signals detected: {len(env.ts_ids)}")
        print(f"Traffic signal IDs: {env.ts_ids}")
        
        # Calculate actual state and action sizes
        state_size = calculate_state_size(env)
        action_size = env.action_space.n        

        
    except Exception as e:
        print(f"Error creating SUMO environment: {e}")
        sys.exit(1)
    
    # Create DDQN agents for each traffic signal
    agents = {}
    for ts_id in env.ts_ids:
        agents[ts_id] = DDQNAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id=ts_id,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq
        )
    
    print(f"Created {len(agents)} DDQN agents")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    
    # Training loop
    for episode in range(args.episodes):
        print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
        
        # Reset environment
        states = env.reset()
        episode_reward = 0
        episode_length = 0
        done = {"__all__": False}
        
        while not done["__all__"]:
            # Get actions from all agents
            actions = {}
            for ts_id in env.ts_ids:
                actions[ts_id] = agents[ts_id].act(states[ts_id], training=True)
            
            # Execute actions in environment
            next_states, rewards, done, info = env.step(actions)
            
            # Store experiences and train agents
            for ts_id in env.ts_ids:
                agents[ts_id].remember(
                    states[ts_id], actions[ts_id], rewards[ts_id], 
                    next_states[ts_id], done.get(ts_id, False)
                )
                agents[ts_id].replay()
            
            # Update metrics
            episode_reward += sum(rewards.values())
            episode_length += 1
            states = next_states
            
            # Progress output
            if args.verbose and episode_length % 100 == 0:
                avg_reward = episode_reward / episode_length / len(agents)
                epsilons = {ts_id: agents[ts_id].epsilon for ts_id in agents.keys()}
                print(f"  Step {episode_length}, Avg reward: {avg_reward:.3f}, Epsilons: {list(epsilons.values())[0]:.3f}")
        
        # Episode completed
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate average loss for this episode
        avg_loss = 0
        for ts_id in agents.keys():
            if agents[ts_id].losses:
                avg_loss += np.mean(agents[ts_id].losses[-1*args.max_steps:])  
        avg_loss /= len(agents)
        training_losses.append(avg_loss)
        
        # Episode summary
        avg_reward_per_agent = episode_reward / len(agents)
        print(f"Episode {episode + 1} completed:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Average reward per agent: {avg_reward_per_agent:.2f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Current epsilon: {agents[list(agents.keys())[0]].epsilon:.3f}")
        
        # Save models periodically
        if (episode + 1) % args.save_freq == 0:
            for ts_id in agents.keys():
                model_path = os.path.join(model_dir, f"ddqn_agent_{ts_id}_episode_{episode + 1}.pt")
                agents[ts_id].save_model(model_path)
            print(f"Models saved at episode {episode + 1}")
        
        # Evaluation episodes (with epsilon=0)
        if (episode + 1) % args.eval_freq == 0:
            print(f"\n--- Evaluation Episode ---")
            eval_states = env.reset()
            eval_reward = 0
            eval_length = 0
            eval_done = {"__all__": False}
            
            while not eval_done["__all__"]:
                eval_actions = {}
                for ts_id in env.ts_ids:
                    eval_actions[ts_id] = agents[ts_id].act(eval_states[ts_id], training=False)
                
                eval_next_states, eval_rewards, eval_done, _ = env.step(eval_actions)
                eval_reward += sum(eval_rewards.values())
                eval_length += 1
                eval_states = eval_next_states
            
            print(f"Evaluation - Total reward: {eval_reward:.2f}, Length: {eval_length}")
    
    # Save final models
    print("\n=== Training Completed ===")
    for ts_id in agents.keys():
        model_path = os.path.join(model_dir, f"ddqn_agent_{ts_id}_final.pt")
        agents[ts_id].save_model(model_path)
    
    # Save training metrics
    metrics_df = pd.DataFrame({
        'episode': range(1, args.episodes + 1),
        'total_reward': episode_rewards,
        'episode_length': episode_lengths,
        'avg_loss': training_losses,
        'avg_reward_per_agent': [r / len(agents) for r in episode_rewards]
    })
    
    metrics_path = os.path.join(output_base_dir, f"training_metrics_{experiment_time}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"Training metrics saved to: {metrics_path}")
    print(f"Final models saved to: {model_dir}")
    print(f"Average reward over last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")
    
    env.close()