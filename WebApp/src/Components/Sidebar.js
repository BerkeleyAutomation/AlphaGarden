import React from "react";
import { slide as Menu } from "react-burger-menu";

export default props => {
  return (
    <Menu {...props}>
      <a className="menu-item" href="/">
        Home
      </a>

      <a className="menu-item" href="/about">
        About
      </a>

      <a className="menu-item" href="/growth">
        Growth
      </a>

      <a className="menu-item" href="/analysis">
        Analysis
      </a>

      <a className="menu-item" href="/simulation">
        Simulation
      </a>

      <a className="menu-item" href="/robot">
        Robot
      </a>

      <a className="menu-item" href="/credits">
        Credits
      </a>
    </Menu>
  );
};
