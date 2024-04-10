import { Separator } from "@/components/ui/separator";
import { useEffect, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import BackgroundPattern from "@/components/ui/background-pattern";

function Main() {
  const cursorSize = 320;

  const mouse = {
    x: useMotionValue(0),

    y: useMotionValue(0),
  };

  const smoothOptions = { damping: 80, stiffness: 200, mass: 0.8 };

  const smoothMouse = {
    x: useSpring(mouse.x, smoothOptions),

    y: useSpring(mouse.y, smoothOptions),
  };

  const manageMouseMove = (e) => {
    const { clientX, clientY } = e;

    mouse.x.set(clientX - cursorSize / 2);

    mouse.y.set(clientY - cursorSize / 2);
  };

  useEffect(() => {
    window.addEventListener("mousemove", manageMouseMove);

    return () => {
      window.removeEventListener("mousemove", manageMouseMove);
    };
  }, []);
  return (
    <div className="flex p-20 2xl:p-40 w-full h-screen">
      <motion.div
        style={{
          left: smoothMouse.x,
          top: smoothMouse.y,
        }}
        className="fixed bg-gray-200/5 rounded-full w-80 h-80 pointer-events-none"
      ></motion.div>
      <BackgroundPattern />
      <div className="bg-black px-20 py-10 rounded-br-2xl w-3/4 h-full">
        <h1 className="font-extrabold font-frank text-7xl text-center text-gray-200">
          CONGONITIVE DRIVER ACTION RECOGNITION SYSTEM
        </h1>
      </div>
      <div className="text-right z-50 flex flex-col gap-10 bg-gray-200 p-10 rounded-2xl w-1/4 2xl:w-[20%] h-full text-black">
        <span className="space-y-4">
          <h2 className="font-extrabold font-libre text-5xl">Team</h2>
          <ul className="space-y-1 font-biryani font-bold text-xl">
            <li>ADITYA ADHANE</li>
            <li>ARYAN GAIKAWAD</li>
            <li>VEDANT MORE</li>
            <li>VISHAL SANGOLE</li>
          </ul>
        </span>
        <Separator className="bg-black" />
        <span className="space-y-4">
          <h2 className="font-extrabold font-libre text-5xl">Guide</h2>
          <p className="font-biryani font-bold text-xl">S. P. JADHAV</p>
        </span>
      </div>
    </div>
  );
}

export default Main;
