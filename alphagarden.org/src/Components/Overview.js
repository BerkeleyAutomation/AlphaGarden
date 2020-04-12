import React from 'react';
import Delayed from './Delayed';
	
 //component for the text that goes over the background video
 const Overview = (props) => {

 	if(props.nuc){setTimeout(props.endFunc, 3000)}
     
	 const formatDate = (date) =>{
		 var monthNames = [
		   "January", "February", "March",
		   "April", "May", "June", "July",
		   "August", "September", "October",
		   "November", "December"
		 ];
		 var d = date.getDate();
		 var m = date.getMonth() + 1;
		 var y = date.getFullYear();
		 return monthNames[m - 1] + ' ' + (d <= 9 ? '0' + d : d) + ', ' + y ;
	 }
 
	 var currentDate = formatDate(new Date());

	 return(
 	 	<div className="LOADING">
			<div id="Overview_Container">
				<img src={require("../Media/Garden-Overview.bmp")} alt="GARDEN" height="100%" width="100%" />
			</div>
			
			<Delayed waitBeforeShow={1000}>
            	<div className="date-box-row">
					<div className="date-box-row-item" id="date-box-div">
						<div className="OverviewDate">
							<p id="jumbotron-date">{currentDate}</p>
						</div>
					</div>
				</div>
			</Delayed>
		</div>
		)    
}

export default Overview;