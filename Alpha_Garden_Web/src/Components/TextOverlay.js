import React from 'react';
	
	let diversity = "36%";
	let plants = ["Lavendar", "Basil", "Turnips", "Rosemary", "Cacti"];
	const plantMap = (x) => {return x + ", "}
 //component for the text that goes over the background video
 const TextOverLay = (props) => {

	let today = new Date();
	 return(
 	 	<div className = "LOADING">
	 		<img src={require("./Garden-Overview.png")} alt="Zaaa GARDEN" height="100%" width="100%" />
	 		<p className="top"> LOADING ALPHA GARDEN... </p>
	    	<div className="Overlay">
		 		<button id="button" onClick={props.endFunc}> Day  {today.getDate() + today.getMonth()} </button>
		 	</div>
		 	<div className = "Data">
		 			<p> Diversity: {diversity} + <br/> Plants: {plants.map(plantMap)} </p>
		 	</div>
		</div>
		)    
}

export default TextOverLay;