import React from 'react';
import Pre_Zoom from './Pre_Zoom'
import Post_Zoom from './Post_Zoom'
import $ from 'jquery'
import PlantData from '../Media/plant-data'
import Overview from './Overview.js'
import { CSSTransition } from 'react-transition-group'
// Component for dynamic zoom data display

class Element3 extends React.Component{

	constructor(props) {

		//Func to trigger the proper transformations to zoom into a square of the garden  Math.floor(Math.random() * 16) + 1
		

		//Func to trigger the proper transformations to zoom into a square of the garden  square=Math.floor(Math.random() * 16) + 1


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
			triggerZoom(square)
			setZoomPosition(square);
			setOverlay(square);

			if(this.props.nuc){
				setTimeout(zoomOut, 14000);
			}
		}

		// Initialize plant data from JSON
		const GARDEN_WIDTH = 1920;
		const GARDEN_HEIGHT = 1080;
		const GRID_WIDTH = 4;
		const GRID_HEIGHT = 4;

		const gridPlants = {};
		for (let i = 1; i <= GRID_WIDTH * GRID_HEIGHT; i++) {
			gridPlants[i] = [];
		}

		for (let plant of PlantData) {
			const { center_x, center_y } = plant;
			const gridX = Math.floor(center_x / (GARDEN_WIDTH / GRID_WIDTH)) + 1;
			const gridY = Math.floor(center_y / (GARDEN_HEIGHT / GRID_HEIGHT));
			const square = gridX + gridY * GRID_WIDTH;
			gridPlants[square].push(plant);
		}

		console.log(gridPlants);

		const removeOverlay = () => {
			this.setState({
				overlay: null
			});
		}

		const triggerZoom = (box) => {
			this.setState({
				zoom: "Zoom" + box,
				handleClick: zoomOut
			});
		}

		const setZoomPosition = (box) => {
			console.log(box)
			console.log(document.getElementById('no_zoom'))
			var img = document.getElementById('no_zoom');
			if (img == null) {
				img = document.getElementById('ZoomOut');
			}
			if (img == null) {
				img = document.getElementById("Zoom" + box);
			}
			var i = box - 1;
			img.style.transformOrigin = (i % GRID_WIDTH) * (100 / (GRID_WIDTH - 1)) + '%' + Math.floor((i / GRID_HEIGHT)) * (100 / (GRID_HEIGHT - 1))  + '%';
			console.log((i % GRID_WIDTH) * (100 / (GRID_WIDTH - 1)) + '%' + Math.floor((i / GRID_HEIGHT)) * (100 / (GRID_HEIGHT - 1))  + '%');
		}

		const setOverlay = (box) => {
			const x = (box - 1) % GRID_WIDTH;
			const y = Math.floor((box - 1) / GRID_WIDTH);

            setTimeout(
            	() => {this.setState({
					overlay: <Post_Zoom box={box} 
								plants={gridPlants[box]} 
								startX={x * (GARDEN_WIDTH / GRID_WIDTH)}
								startY={y * (GARDEN_HEIGHT / GRID_HEIGHT)}
								gridWidth={(GARDEN_WIDTH / GRID_WIDTH)}
								gridHeight={(GARDEN_HEIGHT / GRID_HEIGHT)}
							/>
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

			setTimeout(() => {this.setState({overlay: null})}, 3000);

			if(this.props.nuc){
				this.setState((state) => ({
					counter: state.counter + 1
				}));
				if(this.state.counter >= 2){
					this.props.endFunc();
				}
				else {
					setTimeout(zoomIn, 7000);
				}
			}
		}

    	super(props);

    	this.state = {
    		overlay: null,
    		zoom: "no_zoom",
    		handleClick: zoomIn,
    		x:0,
    		y:0,
    		counter: 0,

    		overview: true
    	}
	}
	
   	componentDidMount(){
   		const timer = () => {
    		this.state.handleClick();
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
					<img src={require("./Garden-Overview.bmp")} alt="Zaaa GARDEN" height="100%" width="100%" onClick={(e) => {console.log("???"); this.state.handleClick(e)}}  id={this.state.zoom}/>
				</div>
				<CSSTransition
		        	in={this.state.overview}
		        	timeout={800}
		        	onEnter={setTimeout(() => this.setState({overview:false}), 5000)}
		        	onExited={() => {
		    		this.state.handleClick();
		   			}}
		        	unmountOnExit
		        	classNames="over"
		        	>
		        	<div>	
       			 <Overview />
       			 </div>
      		</CSSTransition>

				<div className="Overlay">
					{this.state.overlay}
				</div>
		    </div>

	    
	  );
	}
}

export default Element3;