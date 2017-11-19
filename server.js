var mongoose = require('mongoose');
var express = require('express');
var app = express();
var router = require('./router/main.js')(app);

app.set('views',__dirname + '/views');
app.set('view engine','ejs');
app.engine('html', require('ejs').renderFile);
app.use(express.static('public'));

var server = app.listen(2014,function(){
    console.log("Express server has started on port 2011")
})
