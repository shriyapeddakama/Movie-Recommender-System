import argparse
import logging

def main():
    parser = argparse.ArgumentParser(prog="shrek")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start API service")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8082, help="Port to listen on (default: 8082)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info)",
    )

    # Add ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest recent data from Kafka")
    ingest_parser.add_argument(
        "-n", "--num-records",
        type=int,
        default=100000,
        help="Number of recent records to fetch (default: 100000)"
    )
    ingest_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging if verbose
    if hasattr(args, 'verbose') and args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "shrekommender_system.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level,
        )
    elif args.command == "ingest":
        from shrekommender_system.data.ingestion import ingest_recent
        stats = ingest_recent(args.num_records)

        # Print summary
        print("\nIngestion completed:")
        print(f"  Total events: {stats.get('total_events', 0)}")
        print(f"  Watch events: {stats.get('watch_events', 0)}")
        print(f"  Rate events: {stats.get('rate_events', 0)}")
        print(f"  Recommendation events: {stats.get('recommendation_events', 0)}")
        print(f"  Parse errors: {stats.get('parse_errors', 0)}")
        print(f"  Files created: {len(stats.get('files_created', []))}")
    else:
        parser.print_help()
