import {
  createRouter,
  createRoute,
  createRootRoute,
  Outlet,
} from "@tanstack/react-router";
import TestsPage from "./component/TestsPage";
import { TestComponent } from "./component/TestComponent";
import SmokeSimulationComponent from "./component/SmokeSimulationComponent";

// Root route
const rootRoute = createRootRoute({
  component: () => <Outlet />,
});

// Home route
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => <SmokeSimulationComponent />,
});

// Tests listing route
const testsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tests",
  component: TestsPage,
});

// Individual test route
const testRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tests/$testId",
  component: TestComponent,
});

// Create the route tree
const routeTree = rootRoute.addChildren([indexRoute, testsRoute, testRoute]);

// Create the router
export const router = createRouter({ routeTree });

// Register router for typesafety
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
