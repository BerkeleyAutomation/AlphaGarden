import React from 'react';
import GlowingMarks from './GlowingMarks'
	
	let diversity = "36%";
	let growth = "50%";
	let plants = ["Lavendar", "Basil", "Turnips", "Rosemary", "Cacti"];
	const plantMap = (x) => {return x + ", "}
 //component for the text that goes over the background video
 const TextOverLay = (props) => {

	let today = new Date();
	 return(
 	 	<div className = "LOADING">
 	 		<GlowingMarks />
	 		<img src={require("./Garden-Overview.png")} alt="Zaaa GARDEN" height="100%" width="100%" />

	    	<div className="Overlay">
	    		<p className="top"> ALPHA GARDEN</p>
		 		<button id="button" onClick={props.endFunc}> Day  {today.getDate() + today.getMonth()} </button>
		 	</div>
		 	<div className = "Data">
		 			<p> Growth: +{growth} <br/> Diversity: +{diversity} </p>
		 	</div>
		</div>
		)    
}

export default TextOverLay;