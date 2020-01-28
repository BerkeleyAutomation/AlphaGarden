import React from 'react';
import Pre_Zoom from './Pre_Zoom'
import Post_Zoom from './Post_Zoom'
import $ from 'jquery'
// Component for dynamic zoom data display


class Element4 extends React.Component{

	

	constructor(props) {
		//Func to trigger the proper transformations to zoom into a square of the garden
		

		const zoomIn = (square=Math.floor(Math.random() * 16) + 1) => {
			//calculate which square to zoom into
			if(!this.props.nuc){
				if(this.state.x < 0.25){
					square = 1;
				}
				else if( this.state.x < 0.5){
					square = 2;
				}else if(this.state.x < 0.75){
					square = 3;
				}else{
					square = 4;
				}

				if(this.state.y < 0.25){
					square += 0;
				}else if(this.state.y < 0.5){
					square += 4;
				}else if(this.state.y < 0.75){
					square += 8;
				}else if(this.state.y > 0.75){
					square += 12;
				};
		}


			removeOverlay();
			triggerZoom(square);
			setOverlay(square);	

			if(this.props.nuc){
				setTimeout(zoomOut, 6000);
			}
		}




		const removeOverlay = () => {
			this.setState({
				overlay: null
			})
		}

		const triggerZoom = (box) => {
			this.setState({
				zoom: "Zoom" + box,
				handleClick: zoomOut
			})
		

		}

		const setOverlay = (box) => {
            setTimeout(

            	() => {this.setState({
            		overlay: <Post_Zoom box={box}/>
            	})}
            , 3000);
        }
		


		//Func to dynamically zoom back out to the overhead of the garden
		const zoomOut = () => {
			removeOverlay();

			this.setState({
				zoom: "ZoomOut",
				handleClick: zoomIn
			})

			setTimeout(

            	() => {this.setState({
            		overlay: <Pre_Zoom endFunc={this.props.endFunc}/>
            	})}
            , 3000);

           	if(this.props.nuc){
           		setTimeout(zoomIn, 4000);
           	}


		}

    	super(props);

    	this.state = {

    		overlay: <Pre_Zoom endFunc={this.props.endFunc}/>,

    		zoom: "no_zoom",

    		handleClick: zoomIn,

    		x:0,

    		y:0

    	}




    	
    }


  
   	componentDidMount(){
   		const timer = () => {
    		this.state.handleClick();
   		}

    	if(this.props.nuc){
   			setTimeout(timer, 2000);
   		}
    }







    //constantly updates the position of
    _onMouseMove(e) {
    	this.setState({ x: (e.clientX / $( window ).width()), y: (e.clientY / $( window ).height())  });
  	}	

	render(){


	  return (
	  		<div onMouseMove={this._onMouseMove.bind(this)}>
		  	<div id="Zoom_Container">

		    	<img src={require("./Garden-Grid.bmp")} alt="Zaaa GARDEN" height="100%" width="100%" onClick={this.state.handleClick}  id={this.state.zoom}/>

		    	
		    </div>

		    <div className="Overlay">
		    	{this.state.overlay}
		    </div>

		    </div>

	    
	  );
	}
}

export default Element4;