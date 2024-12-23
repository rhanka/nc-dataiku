import dataiku
import pandas as pd
from flask import request
#from flask_cors import CORS


# Example:
# As the Python webapp backend is a Flask app, refer to the Flask
# documentation for more information about how to adapt this
# example to your needs.
# From JavaScript, you can access the defined endpoints using
# getWebAppBackendUrl('nc')

@app.route('/nc')
def first_call():
    max_rows = request.args.get('max_rows') if 'max_rows' in request.args else 500

    mydataset = dataiku.Dataset("NC_types_random_500_final_structured")
    mydataset_df = mydataset.get_dataframe(sampling='head', limit=max_rows)

    mydataset_df['nc_event_date'] = mydataset_df['nc_event_date'].astype(str)
    mydataset_df['analysis_history'] = mydataset_df['analysis_history'].apply(json.loads)

    # Pandas dataFrames are not directly JSON serializable, use to_json()
    data = mydataset_df.to_dict(orient='records')
    return json.dumps(data)
