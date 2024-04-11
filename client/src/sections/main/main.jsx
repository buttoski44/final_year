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
    <>
        <BackgroundPattern />
    <div className="bg-gradient-to-r from-blue-900 flex-grow flex flex-col justify-end  w-full rounded-3xl ">
       <div className="flex justify-center px-20 pt-10  w-full h-screen">
         <motion.div
           style={{
             left: smoothMouse.x,
             top: smoothMouse.y,
            }}
           className="fixed bg-gray-200/5 rounded-full w-80 h-80 pointer-events-none">
         </motion.div>
         
         <div className="flex flex-col items-center">
 <span>
    <h1 className="flex justify-center font-extrabold font-frank text-5xl text-cyan-300 whitespace-nowrap mb-24">
      CONGONITIVE DRIVER ACTION RECOGNITION SYSTEM
    </h1>
 </span>
 <div className="bg-gray-200 text-center flex flex-col gap-12 px-20 py-10 text-black mt-4 rounded-3xl border-4 border-black ring-offset-2 ring">
    <div>
      <h2 className="font-extrabold font-libre text-4xl pb-4">Team</h2>
      <div className="flex justify-center">
        <ul className="text-left space-y-1 font-biryani font-bold text-xl">
          <li className="flex justify-between items-center">
            <span>ADITYA ADHANE</span>
            <span className="ml-10">- 407A004</span>
          </li>
          <li className="flex justify-between items-center">
            <span>ARYAN GAIKAWAD</span>
            <span>- 407A038</span>
          </li>
          <li className="flex justify-between items-center">
            <span>VEDANT MORE</span>
            <span>- 407A075</span>
          </li>
          <li className="flex justify-between items-center">
            <span>VISHAL SANGOLE</span>
            <span>- 407A037</span>
          </li>
        </ul>
      </div>
    </div>
    {/* <Icon icon="fa6-solid:car-side" width="40rem" height="1rem" style={{color: 'black'}} /> */}
    <Separator className="bg-black w-auto" />
    <span>
      <h2 className="font-extrabold font-libre text-4xl">Guide</h2>
      <p className="font-biryani font-bold text-xl pt-4">S. P. JADHAV</p>
    </span>
 </div>
</div>
       </div>
       
    </div>
    </>
   );
}

export default Main;
