import { render, screen } from "@testing-library/react";
import { App } from "./App";

test("renders the dashboard loading shell", () => {
  render(<App />);
  expect(screen.getByText("Loading Dashboard")).toBeInTheDocument();
});
