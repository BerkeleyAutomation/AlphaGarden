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

    // Pick 5 plants at random to highlight
    for (let i = 0; i < plants.length && i < 5; i++) {
      this.displayPlants.push(plants[i]);
    }
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
    return this.displayPlants.map((plant, i) => {
      console.log(plant)
      const [x, y, radius] = this.getPlantCoords(plant);
      console.log("scaled values:", x, y, radius)
      return <Delayed waitBeforeShow={1000 * (i + 1)}><PlantCircle key={i} label={plant.type} x={x} y={y} radius={radius}/></Delayed>
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
        {this.getDateBox()}
      </div>);
  }
}

export default Post_Zoom;