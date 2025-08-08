import { SpeedInsights } from "@vercel/speed-insights/react"
import './App.css'
import AIBusinessPoster from './components/LandingPage'
import NeuralNetworkVisualizer from "./components/NeuralNetworkVisualizer"
import {createBrowserRouter,RouterProvider } from "react-router-dom"

const router = createBrowserRouter([
  {
    path: "/",
    element: <AIBusinessPoster />,
  },
  {
    path: "/visualizer",
    element: <NeuralNetworkVisualizer />,
  },
]);

function App() {

  return (
    <>
      <RouterProvider router={router}/>
      <SpeedInsights />
    </>
  )
}

export default App
