import Main from "./sections/main/main";
import Model from "./sections/model/model";
import Docs from "./sections/docs/docs";
import Footer from "./sections/footer/footer";
import "./App.css";
import { useEffect, useState } from "react";

function getWindowDimensions() {
  const { innerWidth: width, innerHeight: height } = window;
  return {
    width,
    height,
  };
}

function App() {
  const [windowDimensions, setWindowDimensions] = useState(
    getWindowDimensions()
  );

  useEffect(() => {
    function handleResize() {
      setWindowDimensions(getWindowDimensions());
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  if (windowDimensions.width < 1470)
    return (
      <div className="text-center text-white">
        <h1 className="mt-80 font-extrabold font-frank text-6xl">
          PLEASE SWITCH TO DESKTOP
        </h1>
      </div>
    );

  return (
    <div>
      <Main />
      <Model />
      <Docs />
      <Footer />
    </div>
  );
}

export default App;
