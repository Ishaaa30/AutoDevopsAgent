"""
AutoDevOps AI Agent Crew
------------------------
This script launches a multi-agent "hybrid" crew to automate the Pull Request (PR)
code review, testing, and reporting process. It integrates with GitHub for
reading files and posting the final summary.

This version uses the 'gemini-2.5-flash' model for all agents to ensure
compatibility with free-tier API rate limits.
"""
#added comment

# --- 1. Imports ---

# Standard Library
import os
import sys
import subprocess
import tempfile
import logging
import json # For parsing tool outputs

# Third-Party Libraries
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from github import Github, Auth # CRITICAL: Import Auth for modern PAT usage

# --- 2. Configuration and Setup ---

# Configure logging for clear, professional-looking output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# --- Constants ---
# PR_TO_REVIEW is used as a fallback if the GitHub Action environment variable (PR_URL) is missing
PR_TO_REVIEW = "https://github.com/Ishaaa30/AutoDevopsAgent/pull/1"

# LLM Model Selection: Use Flash model for all agents to avoid rate limiting.
LLM_MODEL = "gemini/gemini-2.5-flash"

def setup_environment():
    """
    Loads API keys from the OS environment (Secrets passed by GitHub Actions).
    Validates the keys and sets them for LiteLLM and Tool constructors.
    """
    load_dotenv() # Load local .env variables for local testing fallback
    
    # CRITICAL FIX: Fetch values directly from OS environment (where GitHub Actions puts secrets)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    github_pat = os.getenv("PAT_COMMENT") or os.getenv("GITHUB_PAT") 

    # --- Validation ---
    if not google_api_key:
        logging.error("GOOGLE_API_KEY not found in OS Environment/Secrets.")
        raise ValueError("GOOGLE_API_KEY not found in OS Environment/Secrets.")
    if not github_pat:
        logging.error("GITHUB_PAT/PAT_COMMENT not found in OS Environment/Secrets.")
        raise ValueError("GITHUB_PAT/PAT_COMMENT not found in OS Environment/Secrets.")
    
    # CRITICAL: Set the variables explicitly into os.environ for tools/LiteLLM to access
    os.environ['GOOGLE_API_KEY'] = google_api_key
    os.environ['GITHUB_PAT'] = github_pat 

    logging.info("API Keys loaded successfully and set in OS environment.")
    return github_pat

# --- 3. Custom Agent Tools ---

class GitHubPRTool(BaseTool):
    """Tool to read file contents from a specific GitHub Pull Request."""
    name: str = "GitHub PR File Reader"
    description: str = "Reads all .py files from a given GitHub Pull Request URL."
    github_client: Github = None

    def __init__(self, github_pat: str):
        super().__init__()
        try:
            # FIX: Use Auth.Token() for modern, non-deprecated PyGithub authentication
            auth = Auth.Token(github_pat)
            self.github_client = Github(auth=auth)
            # FIX: Removed auth.get_user().login() which causes 403 Forbidden error in Actions
            logging.info("GitHubPRTool: Authentication client initialized.")
        except Exception as e:
            logging.error(f"GitHubPRTool: Failed to authenticate with PAT. {e}")
            raise Exception(f"Failed to authenticate with GitHub PAT: {e}")

    def _run(self, pr_url: str) -> str:
        """Reads the content of all .py files from the specified GitHub PR."""
        try:
            parts = pr_url.strip("/").split("/")
            if len(parts) < 4 or parts[-2] != "pull":
                return f"Error: Invalid PR URL format. Expected '.../owner/repo/pull/number'."
            
            pr_number = int(parts[-1])
            repo_name = f"{parts[-4]}/{parts[-3]}"
            
            logging.info(f"Tool: Accessing repo '{repo_name}', PR #{pr_number}")

            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            head_sha = pr.head.sha
            
            file_contents = {} # Use a dictionary to store content by filename
            
            for file in pr.get_files():
                if file.filename.endswith(".py"):
                    logging.info(f"Tool: Reading file '{file.filename}'")
                    content_obj = repo.get_contents(file.filename, ref=head_sha)
                    file_content = content_obj.decoded_content.decode('utf-8')
                    file_contents[file.filename] = file_content
            
            if not file_contents:
                return "No .py files were found in this Pull Request."
            
            # Return a JSON string of the files and their content
            return json.dumps(file_contents)

        except Exception as e:
            logging.error(f"Error while processing GitHub PR: {e}")
            return f"Error while processing GitHub PR: {e}"


class RuffTool(BaseTool):
    """
    A custom tool to run the 'ruff' linter on Python code and get a JSON report.
    """
    name: str = "Ruff Linter Tool"
    description: str = "Runs the 'ruff' linter on a given Python code string and returns a JSON report of issues."

    def _run(self, code_content: str) -> str:
        """Runs ruff on the code content."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode='w', encoding='utf-8') as code_file:
                code_file.write(code_content)
                code_file_path = code_file.name # This variable is intentionally unused

            logging.info(f"Tool: Running ruff on {code_file.name}")

            python_executable = sys.executable
            process = subprocess.run(
                [python_executable, "-m", "ruff", "check", code_file.name, "--format=json", "--exit-zero"],
                capture_output=True, text=True, timeout=30
            )
            
            os.unlink(code_file.name)

            if process.stderr:
                logging.warning(f"RuffTool STDERR: {process.stderr}")

            return process.stdout

        except Exception as e:
            if 'code_file' in locals() and os.path.exists(code_file.name):
                os.unlink(code_file.name)
            logging.error(f"Error during ruff execution: {e}")
            return f"Error during ruff execution: {e}"


class BanditTool(BaseTool):
    """
    A custom tool to run the 'bandit' security scanner on Python code.
    """
    name: str = "Bandit Security Tool"
    description: str = "Runs the 'bandit' security scanner on a given Python code string and returns a JSON report of issues."

    def _run(self, code_content: str) -> str:
        """Runs bandit on the code content."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode='w', encoding='utf-8') as code_file:
                code_file.write(code_content)
                code_file_path = code_file.name # This variable is intentionally unused

            logging.info(f"Tool: Running bandit on {code_file.name}")

            python_executable = sys.executable
            process = subprocess.run(
                [python_executable, "-m", "bandit", "-f", "json", "-q", code_file.name],
                capture_output=True, text=True, timeout=30
            )

            os.unlink(code_file.name)
            
            if process.stderr:
                logging.warning(f"BanditTool STDERR: {process.stderr}")

            return process.stdout

        except Exception as e:
            if 'code_file' in locals() and os.path.exists(code_file.name):
                os.unlink(code_file.name)
            logging.error(f"Error during bandit execution: {e}")
            return f"Error during bandit execution: {e}"


class PytestExecutionTool(BaseTool):
    """
    A tool to run pytest *with coverage* on a self-contained test script.
    """
    name: str = "Pytest Coverage Tool"
    description: str = "Saves and runs a self-contained pytest script, capturing the test and coverage report."

    def _run(self, test_code: str) -> str:
        """Runs pytest with pytest-cov and captures the output."""
        test_file_path = None # Define variable in outer scope for cleanup
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="_test.py", mode='w', encoding='utf-8', dir=".") as test_file:
                test_file.write(test_code)
                test_file_path = test_file.name # This variable is intentionally unused

            logging.info(f"Tool: Running tests with coverage from {test_file_path}")

            python_executable = sys.executable
            process = subprocess.run(
                [python_executable, "-m", "pytest", "--cov=.", test_file_path],
                capture_output=True, text=True, timeout=60
            )

            os.unlink(test_file_path)

            if process.returncode == 0:
                return f"All tests passed!\n\nOutput:\n{process.stdout}"
            else:
                return f"Tests FAILED!\n\nSTDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}"

        except Exception as e:
            # FIX: Use the variable test_file_path for cleanup, not the string 'test_file_path'
            if test_file_path and os.path.exists(test_file_path):
                os.unlink(test_file_path)
            logging.error(f"Error during test/coverage execution: {e}")
            return f"Error during test/coverage execution: {e}"


class GitHubPRCommentTool(BaseTool):
    """Tool for posting the final summary comment to a GitHub Pull Request."""
    name: str = "GitHub PR Comment Tool"
    description: str = "Posts a comment on a specific GitHub Pull Request URL."
    github_client: Github = None

    def __init__(self, github_pat: str):
        super().__init__()
        try:
            auth = Auth.Token(github_pat)
            self.github_client = Github(auth=auth)
            logging.info("GitHubPRCommentTool: Authentication client initialized.")
        except Exception as e:
            logging.error(f"GitHubPRCommentTool: Failed to authenticate with PAT. {e}")
            raise Exception(f"Failed to authenticate with GitHub PAT: {e}")

    def _run(self, pr_url: str, comment: str) -> str:
        """Posts a comment to the specified GitHub PR."""
        try:
            parts = pr_url.strip("/").split("/")
            if len(parts) < 4 or parts[-2] != "pull":
                return "Error: Invalid PR URL format."
            
            pr_number = int(parts[-1])
            repo_name = f"{parts[-4]}/{parts[-3]}"
            
            logging.info(f"Tool: Posting comment to repo '{repo_name}', PR #{pr_number}")
            
            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            pr.create_issue_comment(comment)
            
            return "Comment posted successfully."
        except Exception as e:
            logging.error(f"Error posting comment: {e}")
            return f"Error posting comment: {e}"


# --- 4. Main Application Logic ---

def main():
    """
    Main function to set up and run the AutoDevOps Crew.
    """
    
    # --- Setup ---
    try:
        github_pat = setup_environment()
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    pr_url = os.getenv("PR_URL", PR_TO_REVIEW) 

    # --- 1. Instantiate Tools ---
    try:
        github_read_tool = GitHubPRTool(github_pat=github_pat)
        github_comment_tool = GitHubPRCommentTool(github_pat=github_pat) 
    except Exception:
        sys.exit(1) 
    
    pytest_tool = PytestExecutionTool()
    ruff_tool = RuffTool()
    bandit_tool = BanditTool()
    logging.info("All tools instantiated successfully.")

    # --- 2. Define Agents ---
    
    code_reviewer = Agent(
        role='Senior Python Developer',
        goal=(
            "Analyze Python code for style, errors, and logic. Use `ruff` for "
            "speed and apply your expertise for high-level reasoning."
        ),
        backstory=(
            "You are a meticulous Senior Software Engineer. First, use the "
            "`Ruff Linter Tool` to obtain a JSON report of style and bug "
            "issues. Then review that report and the original code to add "
            "human-level feedback on readability and logic. Provide corrected "
            "code blocks only when necessary."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[github_read_tool, ruff_tool],
        llm=LLM_MODEL # <-- Use the single Flash model
    )

    security_agent = Agent(
        role='Python Security Auditor',
        goal=(
            "Analyze code for security vulnerabilities using the `BanditTool`. "
            "Summarize findings with focus on medium and high severity issues."
        ),
        backstory=(
            "You are a DevSecOps specialist. Use the `Bandit Security Tool` to "
            "obtain a JSON report of potential vulnerabilities. Review the "
            "report, ignore low-severity 'Info' items, and summarize any MEDIUM "
            "or HIGH risk issues by file and line number."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[bandit_tool],
        llm=LLM_MODEL # <-- Use the single Flash model
    )

    test_generator = Agent(
        role='Software Quality Assurance Engineer',
        goal=(
            "Generate comprehensive, self-contained pytest unit tests for the "
            "corrected Python code provided by the reviewer."
        ),
        backstory=(
            "You are a skilled QA Engineer. Produce pytest files that are fully "
            "self-contained. Use `@patch` to mock external dependencies for "
            "isolated execution."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=LLM_MODEL # <-- Use the single Flash model
    )

    test_executor = Agent(
        role='Software Test Executor',
        goal='Run the provided self-contained pytest script and report the raw results.',
        backstory=(
            "You are an execution bot. Use the `Pytest Coverage Tool` to run "
            "tests and return the raw output, including the coverage percent."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[pytest_tool],
        llm=LLM_MODEL # <-- Use the single Flash model
    )
    
    report_agent = Agent(
        role='DevOps Reporter',
        goal='Summarize all findings (linting, security, test pass/fail, and coverage) and post them to the GitHub PR.',
        backstory=(
            "You are the final step. Collect all reports, format a single "
            "easy-to-read markdown comment, and post it to the PR."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[github_comment_tool],
        llm=LLM_MODEL # <-- Use the single Flash model
    )
    logging.info("All agents defined.")

    # --- 3. Define Tasks ---
    
    file_reader_task = Task(
        description=(
            f"Use the 'GitHub PR File Reader' tool to read all .py files from the PR: {pr_url}."
        ),
        expected_output=(
            "A JSON string mapping filenames to their full text content."
        ),
        agent=code_reviewer,
    )

    review_task = Task(
        description=(
            "You will receive a JSON string of filenames and code. For *each* file:\n"
            "1. Run the `Ruff Linter Tool` on its code content.\n"
            "2. Parse the ruff JSON output and create a list of findings including"
            " the filename, line number, error code (e.g., F821), and the error"
            " message for each finding.\n"
            "3. Add a high-level review of the file's logic and readability.\n"
            "4. Provide a corrected code block ONLY if changes are necessary."
        ),
        expected_output=(
            "A comprehensive code review report containing:\n"
            "1. A 'Ruff Linter Findings' section formatted as a markdown list"
            " that shows the file, line number, error code, and message for each"
            " issue.\n"
            "2. A 'High-Level Review' section with human-like feedback.\n"
            "3. A SEPARATE Python code block for each corrected file. Each block"
            " must start with: --- START CORRECTED CODE: [filename.py] --- and end"
            " with: --- END CORRECTED CODE: [filename.py] ---."
        ),
        agent=code_reviewer,
        context=[file_reader_task],  # Depends on the file content
    )
    
    security_audit_task = Task(
        description=(
            "You will receive the file content from the first task. For *each* file:\n"
            "1. Run the `Bandit Security Tool` on its code content.\n"
            "2. Parse the JSON report from Bandit.\n"
            "3. Generate a summary of all HIGH or MEDIUM severity issues including"
            " the filename, line number, and the issue text."
        ),
        expected_output=(
            "A security summary titled 'Security Audit Results'.\n"
            "If issues are found, list them by file and line number.\n"
            "If no HIGH or MEDIUM issues are found, state: 'No major security"
            " vulnerabilities found'."
        ),
        agent=security_agent,
        context=[file_reader_task],  # Depends on the file content
    )

    # --- UPDATED TASK: Fixes ModuleNotFoundError ---
    test_generation_task = Task(
        description=(
            "You will receive a review report (which includes corrected code blocks) and a security report. "
            "Your goal is to create a single, runnable pytest script. "
            "\n\n"
            "CRITICAL RULES:\n"
            "1. The test script must be 100% self-contained. You MUST **copy all corrected functions and classes** "
            "(e.g., `GitHubPRTool`, `setup_environment`, etc.) from the 'CORRECTED CODE' blocks "
            "directly into the top of your test script.\n"
            "2. DO NOT try to `import agent_reviewer`, `import sample`, or `import main`. "
            "Only import standard Python libraries like `pytest`, `unittest.mock`, `os`, or `sys`.\n"
            "3. Any pytest fixture you use (like 'mock_client') MUST be defined "
            "with an '@pytest.fixture' decorator in the *same file*. Do not call fixtures "
            "that you have not defined.\n"
        ),
        expected_output=(
            "A single Python code block starting with ```python and ending with ```. "
            "This block must be a complete, runnable pytest file, containing the "
            "**copied source code** at the top, followed by all necessary **fixture definitions**, "
            "followed by the `pytest` test functions."
        ),
        agent=test_generator,
        context=[review_task, security_audit_task]
    )

    test_execution_task = Task(
        description=(
            "You will receive a self-contained pytest script. Use the 'Pytest "
            "Coverage Tool' to run this test code."
        ),
        expected_output=(
            "The full, raw output from the 'Pytest Coverage Tool'. The output"
            " MUST include the test pass/fail summary and the final coverage"
            " percentage (for example, 'Coverage: 92%')."
        ),
        agent=test_executor,
        context=[test_generation_task],
    )
    
    # --- UPDATED TASK: Forces detailed reporting in the final comment ---
    report_task = Task(
        description=(
            "You will receive context from all previous tasks (review, security,"
            " test execution). Create a single, comprehensive summary comment in"
            " markdown for the GitHub PR. Your comment MUST be professionally"
            " formatted and include these sections:\n"
            "1. 'Code Review & Linting': Paste the detailed Ruff Linter Findings"
            " (with line numbers) and the 'High-Level Review' from the review"
            " task.\n"
            "2. 'Security Audit': Paste the detailed Bandit Results (with line"
            " numbers) from the security task.\n"
            "3. 'Test Execution & Coverage': Paste the pytest pass/fail summary"
            " and the final coverage percentage from the test execution task.\n"
            "Once you have the full summary, use the 'GitHub PR Comment Tool' to"
            " post it. The URL to post to is: " + pr_url
        ),
        expected_output=(
            "A confirmation message stating 'Comment posted successfully.'"
        ),
        agent=report_agent,
        context=[review_task, security_audit_task, test_execution_task],
    )
    logging.info("All tasks defined.")

    # --- 4. Create and Run the Crew ---
    code_analysis_crew = Crew(
        agents=[code_reviewer, security_agent, test_generator, test_executor, report_agent],
        tasks=[file_reader_task, review_task, security_audit_task, test_generation_task, test_execution_task, report_task],
        process=Process.sequential,
        verbose=True
    )
    logging.info("Crew created.")
    
    logging.info("Starting Code Analysis Crew...")
    result = code_analysis_crew.kickoff()

    logging.info("Crew Execution Finished!")
    print("\n" + "="*30)
    print(" Final Result (Reporting Status)")
    print("="*30)
    print(result)


if __name__ == "__main__":
    main()