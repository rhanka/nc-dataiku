/*
 * For more information, refer to the "Javascript API" documentation:
 * https://doc.dataiku.com/dss/latest/api/js/index.html
 */


$.getJSON(getWebAppBackendUrl('/nc'), function(data) {
    console.log('Received data from backend', data)
    const output = $('<pre />').text('Backend reply: ' + JSON.stringify(data));
    $('body').append(output)
});
