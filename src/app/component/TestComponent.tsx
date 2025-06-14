import AdvectionTestComponent from "@/lib/simulation/components/tests/AdvectionTestComponent";
import DiffusionTestComponent from "@/lib/simulation/components/tests/DiffusionTestComponent";
import { Link, useParams } from "@tanstack/react-router";
import { Button } from "@/shared/ui/button";
import { ArrowLeft } from "lucide-react";

// Test component that handles the routing logic
export function TestComponent() {
  const { testId } = useParams({ from: "/tests/$testId" });

  // Header component with back button
  const TestHeader = () => (
    <div className="border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/tests" className="flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Tests
            </Link>
          </Button>
          <div className="h-6 w-px bg-border" />
          <h1 className="text-lg font-semibold text-gray-900 capitalize">
            {testId} Test
          </h1>
        </div>
      </div>
    </div>
  );

  const renderTestContent = () => {
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
            <Button variant="outline" asChild>
              <Link to="/tests">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Tests
              </Link>
            </Button>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <TestHeader />
      <div className="container mx-auto">{renderTestContent()}</div>
    </div>
  );
}
