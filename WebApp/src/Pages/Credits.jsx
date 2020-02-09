import React from 'react';
import PageHeading from './PageHeading';
import Watercolor from './Watercolor';

const Credits = () => {
  return (<div>
    <PageHeading title="Alphagarden" subtitle="Credits" />
    <Watercolor type="small" />
    <div className="about-content">
      <p>Can a robot learn to garden?</p>
      <p>AlphaGarden is a robotic artwork that considers natural vs artificial intelligence.</p>

      <h2 className="section-heading">The Alphagarden Collective</h2>
      <table className="credits">
        <tr>
          <td><p>Ken Goldberg</p></td>
          <td><p>Director</p></td>
        </tr>
        <tr>
          <td><p>Yahav Avigal</p></td>
          <td><p>Lead Research and Project Coordination</p></td>
        </tr>
        <tr>
          <td><p>Mark Theis</p></td>
          <td><p>Lead Hardware Engineering</p></td>
        </tr>
        <tr>
          <td><p>Kevin Li</p></td>
          <td><p>Deep Learning and UI Implementation</p></td>
        </tr>
        <tr>
          <td><p>Jensen Gao</p></td>
          <td><p>Deep Learning</p></td>
        </tr>
        <tr>
          <td><p>William Wong</p></td>
          <td><p>Deep Learning</p></td>
        </tr>
        <tr>
          <td><p>Micah Carroll</p></td>
          <td><p>Deep Learning</p></td>
        </tr>
        <tr>
          <td><p>Mark Presten</p></td>
          <td><p>Hardware Engineering</p></td>
        </tr>
        <tr>
          <td><p>Grady Pierroz</p></td>
          <td><p>Plant Biology</p></td>
        </tr>
        <tr>
          <td><p>Shivin Devgon</p></td>
          <td><p>Plant Modeling</p></td>
        </tr>
        <tr>
          <td><p>Shubha Jagannatha</p></td>
          <td><p>Plant Modeling</p></td>
        </tr>
        <tr>
          <td><p>Maya Man</p></td>
          <td><p>UI Design</p></td>
        </tr>
        <tr>
          <td><p>Isaac Blankensmith</p></td>
          <td><p>UI Design</p></td>
        </tr>
        <tr>
          <td><p>Sona Dolasia</p></td>
          <td><p>UI Design</p></td>
        </tr>
        <tr>
          <td><p>Mark Selden</p></td>
          <td><p>UI Implementation</p></td>
        </tr>
        <tr>
          <td><p>Jeff He</p></td>
          <td><p>UI Implementation</p></td>
        </tr>
        <tr>
          <td><p>Jackson Chui</p></td>
          <td><p>UI Implementation</p></td>
        </tr>
      </table>

      <h2 className="section-heading">Advisors</h2>
      <table className="credits">
        <tr>
          <td><p>Erig Siegel</p></td>
          <td><p>Plant Biology</p></td>
        </tr>
        <tr>
          <td><p>Brian Bailey</p></td>
          <td><p>Plant Biology</p></td>
        </tr>
        <tr>
          <td><p>Sarah Newman</p></td>
          <td><p>Media Art Contextualization</p></td>
        </tr>
        <tr>
          <td><p>Gil Gershoni</p></td>
          <td><p>Design Consultant</p></td>
        </tr>
        <tr>
          <td><p>Stavros Vougioukas</p></td>
          <td><p>Agricultural Robotics</p></td>
        </tr>
        <tr>
          <td><p>Stefano Carpin</p></td>
          <td><p>Agricultural Robotics</p></td>
        </tr>
        <tr>
          <td><p>Joshua Viera</p></td>
          <td><p>Water Engineering</p></td>
        </tr>
        <tr>
          <td><p>Sonia Uppal</p></td>
          <td><p>UI Implementation</p></td>
        </tr>
        <tr>
          <td><p>Jeff Ichnowski</p></td>
          <td><p>UI Implementation</p></td>
        </tr>
      </table>

      <h2 className="section-heading">Acknowledgements</h2>
      <p>Watercolor by Chelsea Qiu and Sarah Newman</p>
      <p>UC Berkeley AUTOLAB</p>
      <p>NSF-USDA Grant: RAPID: Robot-Assisted Precision Irrigation Delivery</p>
    </div>
  </div>)
}

export default Credits;