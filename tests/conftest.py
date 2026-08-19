import pytest

SAMPLE_RESUME_TEXT = """Jane Doe

Education
B.Sc. in Computer Science, Chulalongkorn University

Experience
Data Analyst Intern, ABC Corp
Built ETL pipelines in Python and SQL.

Skills
Python, SQL, Pandas, scikit-learn"""


@pytest.fixture
def sample_resume_pdf(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in SAMPLE_RESUME_TEXT.split("\n"):
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    path = tmp_path / "sample_resume.pdf"
    pdf.output(str(path))
    return path


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "UPLOAD_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def disable_auth_by_default(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "api_auth_token", "")
