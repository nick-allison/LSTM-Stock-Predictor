from model_functions import train_model, test_model

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train or test the LSTM model.')
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--ticker', type=str, help='Ticker symbol for testing')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON files folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save or load the model')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for specific stock during testing')

    args = parser.parse_args()

    if args.mode == 'train':
        # Train the model
        train_model(args.json_path, args.model_path)
    elif args.mode == 'test':
        if not args.ticker:
            print("Please provide a ticker symbol for testing using --ticker.")
            return
        # Test the model on the specified ticker
        test_model(args.ticker.upper(), args.json_path, args.model_path, optimize=args.optimize)

if __name__ == "__main__":
    main()
