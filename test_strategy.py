from strategy import RankingStrategy

strategy = RankingStrategy()

df = strategy.load_data("multi_stock_dataset.csv")

df = strategy.normalize_features(df)

strategy.train(df)

df = strategy.predict(df)

df = strategy.generate_signals(df)

df = strategy.position_sizing(df)

returns = strategy.compute_returns(df)

print("Sharpe:", strategy.sharpe_ratio(returns))