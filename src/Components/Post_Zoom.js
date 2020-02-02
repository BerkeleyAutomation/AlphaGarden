import React from 'react';
import Delayed from './Delayed';
import PlantCircle from './PlantCircle';
import $ from 'jquery';
import DateBox from './DateBox';

class Post_Zoom extends React.Component {
  constructor(props) {
    super(props);

    const { plants } = props;
    this.displayPlants = [];

    // Shuffle array
    for (let i = plants.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [plants[i], plants[j]] = [plants[j], plants[i]];
    }

    // Pick 3 plants at random to highlight
    let k = 0;
    while (k < plants.length && k < 3) {
      if (this.withinBounds(plants[k])) {
        this.displayPlants.push(plants[k]);
      }
      k++;
    }
  }

  withinBounds(plant) {
    const {startX, startY, gridWidth, gridHeight} = this.props;
    const plantX1 = Math.max(plant.center_x - plant.radius, startX);
    const plantY1 = Math.max(plant.center_y - plant.radius, startY);
    const plantX2 = Math.min(plant.center_x + plant.radius, startX + gridWidth);
    const plantY2 = Math.min(plant.center_y + plant.radius, startY + gridHeight);
    
    const visibleArea = (plantX2 - plantX1) * (plantY2 - plantY1);
    const totalArea = plant.radius * plant.radius * 4;
    if (visibleArea / totalArea > 0.7) {
      console.log(visibleArea, totalArea);
      console.log(plantX1, plantX2, plantY1, plantY2)
      console.log("Accepting plant", plant, visibleArea / totalArea);
    }
    return visibleArea / totalArea > 0.7;
  }

  getPlantCoords(plant) {
    const windowWidth = $(window).width();
    const windowHeight = $(window).height();

    let {center_x, center_y, radius} = plant;
    const {startX, startY, gridWidth, gridHeight} = this.props;

    console.log("Before conversion:", center_x, center_y)
    console.log("window size:", $(window).width(), $(window).height());
    console.log("grid values:", startX, startY, gridWidth, gridHeight);
    // Shift x and y
    center_x -= startX;
    center_y -= startY;

    console.log("shifted x, y:", center_x, center_y);

    // Scale x and y and radius to match screen dimensions
    // console.log(windowWidth);
    // console.log(gridWidth);
    center_x *= (windowWidth / gridWidth);
    center_y *= (windowHeight / gridHeight);
    radius *= (windowWidth / gridWidth)
    return [center_x, center_y, radius]
  }

  getPlantCircles() {
    const {startX, startY, gridWidth, gridHeight} = this.props;
    return this.displayPlants.map((plant, i) => {
      console.log(plant)
      const [x, y, radius] = this.getPlantCoords(plant);
      console.log("scaled values:", x, y, radius)
      return <Delayed waitBeforeShow={2000 * (i + 1)}><PlantCircle key={i} label={plant.type} x={x} y={y} radius={radius} 
        {...[startX, startY, gridWidth, gridHeight]}/></Delayed>
    })
  }

  getDateBox() {
    const windowHeight = $(window).height();
    return <DateBox x={40} y={windowHeight - 80}/>
  }

  render() {
    return (
      <div className="Overlay">
        {/* <h1> {props.box} </h1> */}
        {this.getPlantCircles()}
        {/* {this.getDateBox()} */}
      </div>);
  }
}

export default Post_Zoom;