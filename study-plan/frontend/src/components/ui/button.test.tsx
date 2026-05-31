import { render, screen } from "@testing-library/react";
import { Button } from "./button";

test("renders a shadcn button", () => {
  render(<Button>Refresh</Button>);
  expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
});
