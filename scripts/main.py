import argparse

from evaluate_model import evaluate_model
from predict_violations import predict_violations, get_violation_impacts
from preprocess import preprocess_data
from train_model import train_model


def main():
	parser = argparse.ArgumentParser(
		description="Run the accessibility ML pipeline."
	)
	parser.add_argument(
		"--steps",
		nargs="*",
		default=["preprocess", "train", "evaluate"],
		choices=["preprocess", "train", "evaluate", "predict"],
		help="Pipeline steps to run in order.",
	)
	parser.add_argument(
		"--predict-url",
		help="Optional URL to run prediction after the pipeline.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=3,
		help="Number of top predictions to display for a URL.",
	)
	parser.add_argument(
		"--min-confidence",
		type=float,
		default=0.0,
		help="Filter predictions below this probability.",
	)
	parser.add_argument(
		"--prior-k",
		type=int,
		default=5,
		help="Number of category-based prior suggestions to show.",
	)
	parser.add_argument(
		"--debug-scrape",
		action="store_true",
		help="Print scraped metadata used for prediction.",
	)
	parser.add_argument(
		"--summary-only",
		action="store_true",
		help="Print only accuracy and F1 summaries for train/test.",
	)
	parser.add_argument(
		"--bootstrap-eval",
		action="store_true",
		help="Compute bootstrap confidence intervals during evaluation.",
	)
	parser.add_argument(
		"--bootstrap-samples",
		type=int,
		default=1000,
		help="Number of bootstrap samples to draw for evaluation.",
	)
	parser.add_argument(
		"--bootstrap-alpha",
		type=float,
		default=0.05,
		help="Alpha for bootstrap confidence intervals (e.g., 0.05 = 95% CI).",
	)
	parser.add_argument(
		"--bootstrap-seed",
		type=int,
		default=42,
		help="Random seed for bootstrap sampling.",
	)
	parser.add_argument(
		"--min-label-count",
		type=int,
		default=20,
		help="Minimum label frequency to keep; others map to 'Other'.",
	)
	args = parser.parse_args()

	if "preprocess" in args.steps:
		preprocess_data()
	if "train" in args.steps:
		train_model(
			summary_only=args.summary_only,
			min_label_count=args.min_label_count,
		)
	if "evaluate" in args.steps:
		evaluate_model(
			bootstrap=args.bootstrap_eval,
			n_boot=args.bootstrap_samples,
			alpha=args.bootstrap_alpha,
			seed=args.bootstrap_seed,
		)

	should_predict = "predict" in args.steps or args.predict_url is not None
	if should_predict:
		if not args.predict_url:
			raise SystemExit("--predict-url is required when using --steps predict")
		result = predict_violations(
			args.predict_url,
			top_k=args.top_k,
			min_confidence=args.min_confidence,
			debug=args.debug_scrape,
			prior_k=args.prior_k,
		)
		print(f"Top {args.top_k} predictions (min={args.min_confidence:.2f}):")
		predictions = result["predictions"]
		if not predictions:
			print("No predictions above threshold.")
		impact_lines = []
		for label, score in predictions:
			print(f"- {label}: {score:.3f}")
			impacts = get_violation_impacts(label)
			if impacts:
				impact_lines.append((label, impacts))

		if impact_lines:
			print("Affected users by predicted issues:")
			for label, impacts in impact_lines:
				print(f"- {label}:")
				for impact in impacts:
					print(f"  - {impact}")

		priors = result["priors"]
		if priors:
			print("Common issues for this category:")
			for name, count, share in priors:
				print(f"- {name}: {share:.0%} of sampled issues (n={count})")


if __name__ == "__main__":
	main()
