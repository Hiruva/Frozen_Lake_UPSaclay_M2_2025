import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Paramètres Q-Learning
is_train = False  # True pour l'entraînement, False pour la phase de test
alpha = 0.1  # Taux d'apprentissage
gamma = 0.95  # Facteur de réduction des récompenses futures
epsilon = 1.0  # Probabilité d'exploration (ε-greedy)
num_episodes = 10000  # Nombre d'épisodes d'entraînement
num_test_episodes = 10  # Nombre d'épisodes de test
render = True  # Activer le rendu uniquement pour la phase de test

# Créer l'environnement sans rendu pour l'entraînement
env_train = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode=None)

# Créer l'environnement avec rendu pour les tests
env_test = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human')

n_states = env_train.observation_space.n
n_actions = env_train.action_space.n

# Initialisation de la table Q
Q = np.zeros((n_states, n_actions))
train_rewards_per_episode = []

# Fonction ε-greedy pour choisir une action
def epsilon_greedy_policy(state, epsilon, training=True):
    if training and np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Exploration
    return np.argmax(Q[state, :])  # Exploitation

# Phase d'entraînement
if is_train:
    for episode in range(num_episodes):
        state, _ = env_train.reset()
        total_reward = 0
        done = False

        while not done:
            # Choisir une action selon la politique ε-greedy
            action = epsilon_greedy_policy(state, epsilon)
            
            # Effectuer l'action et observer le résultat
            next_state, reward, done, truncated, _ = env_train.step(action)

            # Mise à jour de Q avec Q-Learning
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )

            state = next_state  # Mettre à jour l'état
            total_reward += reward

        train_rewards_per_episode.append(total_reward)
        epsilon = max(0.1, epsilon * 0.99)  # Réduction progressive d'epsilon

        if episode % 100 == 0:
            avg_reward = np.mean(train_rewards_per_episode[-100:])
            print(f"Épisode {episode}, Récompense moyenne (100 derniers) : {avg_reward:.2f}")

    # Sauvegarder la table Q après l'entraînement
    np.save("Q_table.npy", Q)
    print("Table Q sauvegardée dans 'Q_table.npy'.")

    # Calcul de la moyenne glissante
    window_size = 100
    if len(train_rewards_per_episode) >= window_size:
        rolling_avg = np.convolve(train_rewards_per_episode, np.ones(window_size) / window_size, mode="valid")
    else:
        rolling_avg = []

    # Tracer les courbes d'entraînement
    plt.figure(figsize=(12, 6))
    plt.plot(train_rewards_per_episode, label="Récompenses par épisode")
    if len(rolling_avg) > 0:
        plt.plot(range(window_size - 1, len(train_rewards_per_episode)), rolling_avg, label="Moyenne glissante (100)")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompenses cumulées")
    plt.title("Entraînement avec Q-Learning sur FrozenLake")
    plt.legend()
    plt.show()

# Phase de test
else:
    # Charger la table Q pour le test
    Q = np.load("Q_table.npy")
    print("Table Q chargée depuis 'Q_table.npy'.")

    test_rewards_per_episode = []

    for episode in range(num_test_episodes):
        state, _ = env_test.reset()
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy_policy(state, epsilon=0, training=False)  # Toujours greedy pendant le test
            state, reward, done, truncated, _ = env_test.step(action)
            total_reward += reward

        test_rewards_per_episode.append(total_reward)

    # Résultats des tests
    average_test_reward = np.mean(test_rewards_per_episode)
    print(f"Récompense moyenne sur {num_test_episodes} épisodes de test : {average_test_reward:.2f}")
    print(f"Taux de succès : {average_test_reward * 100:.2f}%")

    # Distribution des récompenses de test
    plt.figure(figsize=(10, 5))
    plt.hist(test_rewards_per_episode, bins=3, rwidth=0.8, alpha=0.7)
    plt.xlabel("Récompense obtenue")
    plt.ylabel("Nombre d'épisodes")
    plt.title("Distribution des récompenses pendant la phase de test")
    plt.show()
