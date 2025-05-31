import {
  createRouter,
  createRoute,
  createRootRoute,
  Link,
  Outlet,
  useParams,
} from "@tanstack/react-router";
import SmokeSimulationComponent from "./simulation/SmokeSimulationComponent";
import TestsPage from "./pages/TestsPage";
import AdvectionTestComponent from "./components/tests/AdvectionTestComponent";
import DiffusionTestComponent from "./components/tests/DiffusionTestComponent";

// Root route
const rootRoute = createRootRoute({
  component: () => <Outlet />,
});

// Home route
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => (
    <div className="p-8">
      <SmokeSimulationComponent />
    </div>
  ),
});

// Tests listing route
const testsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tests",
  component: TestsPage,
});

// Test component that handles the routing logic
// eslint-disable-next-line react-refresh/only-export-components
function TestComponent() {
  const { testId } = useParams({ from: "/tests/$testId" });

  switch (testId) {
    case "advection":
      return <AdvectionTestComponent />;
    case "diffusion":
      return <DiffusionTestComponent />;
    default:
      return (
        <div className="p-8 text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Test Not Found
          </h2>
          <p className="text-gray-600 mb-6">
            The test "{testId}" could not be found.
          </p>
          <Link
            to="/tests"
            className="text-blue-600 hover:text-blue-800 font-medium"
          >
            ← Back to Tests
          </Link>
        </div>
      );
  }
}

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
