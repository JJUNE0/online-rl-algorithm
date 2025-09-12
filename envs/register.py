

def make(env_id):
    # Prebuilt Environments (OpenAI Gym's MuJoCo)
    if env_id.startswith('Ant') or env_id.startswith('HalfCheetah') or env_id.startswith('Hopper') \
            or env_id.startswith('Humanoid') or env_id.startswith('Swimmer') or env_id.startswith('InvertedPendulum') \
            or env_id.startswith('Reacher') or env_id.startswith('Pusher'):
        return OpenAIGym(env_id=env_id)