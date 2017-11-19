var mongoose = require("mongoose");
var Schema = mongoose.Schema;

var multiparty = require("multiparty");
var multer = require("multer")
var fs = require("fs");
var bodyParser = require('body-parser');
var cmd = require('node-cmd');
var randomToken = require('random-token');

var sessions = [];
var total_count = 1;

function checkSessionAndGetIndex(token){
	for(var i = 0; i <sessions.length; i++){
		if(sessions[i].token == token){
			return i;
		}
	}
	return -1;
}

/*
	@0831 LJW
	type = {
	    token : TOKEN_OF_SESSION,
	    img_path : IMG_PATH_WHICH_USER_UPLOADED
	}
*/
function pushSession(token){
	var tmptype ={
	    token : token,
	    img_path : "",
	    s_time : 0
	}
	sessions.push(tmptype);
	return token;
}

function updateSessionWithType(type){
	var session_idx = checkSessionAndGetIndex(type.token);
	if(session_idx==-1) return -1;
	console.log(session_idx);
	console.log(sessions[session_idx]);
	sessions[session_idx] = type;
}

function expireSession(token){
	console.log("current sessions : ");
	console.log(sessions);
	console.log(token);
	setTimeout(function(){
		var tmpidx = checkSessionAndGetIndex(token);
		console.log(token);
		console.log(sessions[tmpidx]);
		if(sessions[tmpidx].s_time==0){
			console.log("expired session :");
			console.log(sessions.splice(checkSessionAndGetIndex(token),1));
		}else{
			console.log("session's expire time had been postponed.");
			sessions[tmpidx].s_time = 0;
			expireSession(token);
		}
		console.log("remain sessions : ");
		console.log(sessions);
	},1000000);
}


mongoose.Promise = require("bluebird");
/* mongodb connection */

var db = mongoose.connection;
db.on('error', console.error);
db.once('open', () => { console.log('Connected to mongodb server'); });

mongoose.connect('mongodb://localhost/fookey');

var foodSchema = new Schema({
	food_name : String,
	food_category : String,
	calorie : String,
	safety : String,
	nutrient : { carbohydrate : Number, protein : Number , fat : Number}
});

var Food = mongoose.model("foods",foodSchema);

module.exports = function(app) {
  app.get('/', function(req, res) {
    res.render('index', {
      title: "Classification Prototype",
      isUp: false,
      imgs: "./imgs/ml_icons.jpg"
    })
  })


  app.post('/upload', multer({
    dest: 'public/myfiletmp/upload/'
  }).single('upfile'), function(req, res) {
    console.log("we running python...");
    var pyProcess = cmd.get("python public/fookey1/singlecl_fc_cnn.py ../myfiletmp/upload/" + req.file.filename, function(data, err, stderr) {
      if (!err) {
        console.log("data from python script" + data)
        res.render('index', {
          title: data,
          isUp: true,
          imgs: "myfiletmp/upload/" + req.file.filename
        })
      } else {
        console.log("python script cmd error : " + err)
        res.render('index', {
          title: err,
          isUp: true,
          imgs: "myfiletmp/upload/" + req.file.filename
        })
      }
    });
  })

  app.get('/get_token', function(req, res){
	var token = randomToken(16);
	expireSession(pushSession(token));
	console.log("we send token : "+token);
	res.status(200).send({token : token});
  });

  app.post('/img_upload', function(req, res) {
      var user_token = sessions[0].token;
      var form = new multiparty.Form();
      var tres = {};
      var fileName = '';
      var target_path;
      var tmp_path;
      form.on("field",function(name,value){
	user_token = value;
      });
      form.on("file", function(name, file) {
	fileName = file.originalFilename;
	console.log("fileName: " + file.originalFilename);
        console.log("dirname: "+__dirname);
        target_path = __dirname + '/../public/uploads/' + file.path;
	console.log("target path: "+target_path);
	tmp_path = file.path;
	console.log("tmp_path :"+tmp_path);
        fs.renameSync(tmp_path, target_path, function(err) {
	  console.log("rename @@ :");
          if (err) console.error(err.stack);
        });
	var pyProcess = cmd.get("python public/fookey1/singlecl_fc_cnn_first_net.py ../uploads/" +file.path , function(data, err, stderr) {
          if (!err) {
              console.log("data from python script" + data);
          } else {
            console.log("python script cmd error : " + err);
	    console.log(err[0]);
	    var tmptype ={
		token : user_token,
		img_path : file.path,
		s_time : 1
	    }
	    var re = /\s*;\s*/;
	    var tmparray = err.split(re);
	    if(updateSessionWithType(tmptype)!=-1){
		console.log("we send : "+tmparray);
    	        res.status(200).send({
		    categories : tmparray
	        });
	    }else{
		res.status(200).send({
		    msg : "your session has been expired"
		});
	    }
          }
          });
      });
      form.parse(req);
  });

  app.get("/select_main/:category", function(req, res) {
    var session_idx = checkSessionAndGetIndex(req.query.token);
    console.log("select_main, user_token : " + req.query.token);
    console.log(session_idx +": " + sessions[session_idx].img_path);
    if(session_idx == -1){
	res.status(200).send({
	    msg : "your session has been expired"
	});
	return;
    }
    var tmptype ={
	token : sessions[session_idx].token,
	img_path : sessions[session_idx].img_path,
	s_time : 1,
    }

    var pyProcess = cmd.get("python public/fookey1/single_fc_cnn_second_net.py ../uploads/"+tmptype.img_path+" "+req.params.category, function(data,err,stderr){
	if(!err){
	    console.log("data from python script"+ data);
	}else{
	    var re = /\s*;\s*/;
	    var tmparray = err.split(re);
	    console.log("python script cmd error : " +tmparray);
	    res.status(200).send({
		categories : tmparray
	    });
	}
    });
  });
  app.get("/test",function(req,res){
	var food = new Food({
	   food_name : "MixedRice",
	   food_category : "한식,밥",
	   calorie : "13.9kcal / 8장(3.5g)",
	   safety : "안전식품",
	   nutrient : {carbohydrate : 0 , protein : 0, fat : 0}
	});
	food.save(function(err,food){
	   if(err) return console.log(err);
	   console.log("insert successful : " + food);
	});
	Food.findOne({food_name:"계란후라이"}).exec(function(err,doc){
	   console.log(doc);
	   res.status(200).json(doc);
	});
  });

  app.get("/select_sub/:category", function(req, res) {
    /*
    var i;
    var tmpsession;
    for(i = 0 ; i < sessions.length ; i++){
      if(sessions[i].session = req.sessionID){
        tmpsession = sessions[i];
      }
    }
    console.log(req);
    tres = {
      msg: "suscces",
      prm: req.params.cate_sub,
      session : tmpsession
    }
    */
    console.log("params category : "+ req.params.category);
    Food.findOne({food_name: req.params.category}).exec(function(err,doc){
       console.log(doc);
       res.status(200).send({food_info : [
           doc.food_name,
	   doc.food_category,
	   doc.calorie,
	   doc.safety,
	   doc.nutrient.carbohydrate.toString(),
	   doc.nutrient.protein.toString(),
	   doc.nutrient.fat.toString()
       ]})
       //res.status(200).send({description : doc});
    });
  });
}
