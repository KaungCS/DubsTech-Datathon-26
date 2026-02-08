from flask import Flask, render_template, request, jsonify
from predict_violations import predict_violations
from web_scraper import scrape_metadata
import os

# Get absolute path to tableau folder
tableau_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tableau')

app = Flask(__name__, template_folder=tableau_folder, static_folder=tableau_folder, static_url_path='')

@app.route('/')
def index():
    return render_template('data_vis.html')

@app.route('/ml')
def ml():
    return render_template('ml.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to predict violations for a URL."""
    try:
        data = request.json
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Ensure URL has protocol
        if not url.startswith('http'):
            url = 'https://' + url
        
        domain_category = ""
        try:
            metadata = scrape_metadata(url)
            domain_category = metadata.get("domain_category", "")
        except Exception:
            domain_category = ""

        # Run prediction
        result = predict_violations(
            url,
            top_k=3,
            min_confidence=0.0,
            debug=False,
            prior_k=5,
        )
        
        predictions = result['predictions']
        priors = result['priors']
        
        # Format predictions with affected users
        from predict_violations import get_violation_impacts
        formatted_predictions = []
        for label, score in predictions:
            impacts = get_violation_impacts(label)
            formatted_predictions.append({
                'label': label,
                'score': round(score, 3),
                'affected_users': impacts
            })
        
        formatted_priors = [
            {
                'name': name,
                'count': count,
                'share': round(share * 100, 1)
            }
            for name, count, share in priors
        ]
        
        return jsonify({
            'success': True,
            'url': url,
            'domain_category': domain_category,
            'predictions': formatted_predictions,
            'priors': formatted_priors
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print(f'Starting server on 0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=debug)

    app.run(host='0.0.0.0', port=port, debug=debug)
