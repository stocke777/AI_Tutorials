# AI-Powered Email Marketing Bot - Clean & Practical Implementation
# Focus on real AI value with minimal, necessary dependencies

import pandas as pd
import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List
import openai
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import Counter

class AIEmailBot:
    def __init__(self, openai_api_key: str):
        """Initialize AI email bot with OpenAI API key"""
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.db_path = "ai_email_bot.db"
        self.setup_database()
        
    def setup_database(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                company TEXT,
                industry TEXT,
                engagement_score REAL DEFAULT 0.5,
                last_open_time TEXT,
                preferred_tone TEXT DEFAULT 'professional',
                response_pattern TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                content TEXT NOT NULL,
                ai_insights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                emails_sent INTEGER DEFAULT 0,
                opens INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_id INTEGER,
                campaign_id INTEGER,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                opened BOOLEAN DEFAULT FALSE,
                clicked BOOLEAN DEFAULT FALSE,
                send_hour INTEGER,
                FOREIGN KEY (recipient_id) REFERENCES recipients (id),
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def add_recipients(self, recipients: List[Dict]):
        """Add recipients to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for recipient in recipients:
            cursor.execute('''
                INSERT OR REPLACE INTO recipients (email, name, company, industry)
                VALUES (?, ?, ?, ?)
            ''', (
                recipient['email'],
                recipient.get('name', ''),
                recipient.get('company', ''),
                recipient.get('industry', '')
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Added {len(recipients)} recipients")

    def analyze_recipient_behavior(self, recipient_email: str) -> Dict:
        """Analyze individual recipient behavior using AI"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recipient's email history
        cursor.execute('''
            SELECT eh.opened, eh.clicked, eh.send_hour, c.subject, c.content
            FROM email_history eh
            JOIN campaigns c ON eh.campaign_id = c.id
            JOIN recipients r ON eh.recipient_id = r.id
            WHERE r.email = ?
            ORDER BY eh.sent_at DESC
            LIMIT 10
        ''', (recipient_email,))
        
        history = cursor.fetchall()
        conn.close()
        
        print(history)
        if not history:
            return {
                "engagement_score": 0.5,
                "preferred_tone": "professional",
                "best_send_hour": 10,
                "response_pattern": "new_subscriber"
            }
        
        # Calculate basic metrics
        opens = sum(1 for h in history if h[0])
        clicks = sum(1 for h in history if h[1])
        total_emails = len(history)
        
        # Find best send hours
        successful_hours = [h[2] for h in history if h[0]]  # Hours when they opened
        best_hour = Counter(successful_hours).most_common(1)[0][0] if successful_hours else 10
        
        # Get recent subjects for AI analysis
        recent_subjects = [h[3] for h in history[-5:] if h[3]]
        
        # Use AI to analyze tone preferences
        if recent_subjects:
            tone_analysis = self._analyze_tone_preference(recent_subjects, opens/total_emails)
        else:
            tone_analysis = "professional"
        
        return {
            "engagement_score": (opens/total_emails) * 0.7 + (clicks/total_emails) * 0.3,
            "preferred_tone": tone_analysis,
            "best_send_hour": best_hour,
            "response_pattern": "engaged" if opens/total_emails > 0.3 else "low_engagement"
        }

    def _analyze_tone_preference(self, subjects: List[str], open_rate: float) -> str:
        """Use AI to determine preferred communication tone"""
        
        prompt = f"""
        Analyze these email subjects to determine the recipient's preferred communication tone:
        
        Recent subjects: {subjects}
        Open rate: {open_rate:.2%}
        
        Based on which subjects they responded to, determine their preferred tone:
        - professional: Formal, business-focused
        - casual: Friendly, conversational
        - urgent: Time-sensitive, action-oriented
        - informative: Educational, detailed
        
        Return only one word: professional, casual, urgent, or informative
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an email marketing analyst. Analyze communication preferences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            tone = response.choices[0].message.content.strip().lower()
            valid_tones = ["professional", "casual", "urgent", "informative"]
            return tone if tone in valid_tones else "professional"
            
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return "professional"

    def generate_personalized_email(self, 
                                   recipient_email: str,
                                   campaign_purpose: str,
                                   product_info: str,
                                   key_benefits: List[str] = None) -> Dict:
        """Generate AI-personalized email based on recipient behavior"""
        
        # Get recipient data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM recipients WHERE email = ?', (recipient_email,))
        recipient = cursor.fetchone()
        conn.close()
        
        if not recipient:
            return {"error": "Recipient not found"}
        
        # Get AI behavioral insights
        behavior = self.analyze_recipient_behavior(recipient_email)
        
        # Build personalization context
        context = {
            "name": recipient[2] or "there",
            "company": recipient[3] or "your company",
            "industry": recipient[4] or "your industry",
            "engagement_score": behavior["engagement_score"],
            "preferred_tone": behavior["preferred_tone"],
            "response_pattern": behavior["response_pattern"]
        }
        
        # Generate personalized content
        prompt = f"""
        Create a personalized email for this specific recipient:
        
        RECIPIENT PROFILE:
        - Name: {context['name']}
        - Company: {context['company']}
        - Industry: {context['industry']}
        - Engagement Score: {context['engagement_score']:.2f}
        - Preferred Tone: {context['preferred_tone']}
        - Response Pattern: {context['response_pattern']}
        
        CAMPAIGN DETAILS:
        - Purpose: {campaign_purpose}
        - Product: {product_info}
        - Key Benefits: {key_benefits or ['Not specified']}
        
        PERSONALIZATION REQUIREMENTS:
        1. Use their preferred tone ({context['preferred_tone']})
        2. Address them by name naturally
        3. Reference their company/industry meaningfully
        4. If low engagement, use re-engagement tactics
        5. Match content length to their response pattern
        
        Create:
        1. Subject line optimized for their preferences
        2. Email body (HTML format)
        3. Call-to-action that matches their engagement level
        
        Return as JSON:
        {{
            "subject": "subject line here",
            "body": "HTML email body here",
            "personalization_notes": "brief explanation of personalization strategy"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI email personalization expert. Create emails that feel individually written."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            print(response.choices[0].message.content)
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            return {
                "subject": f"Hi {context['name']}, important update about {product_info}",
                "body": f"<p>Hi {context['name']},</p><p>I wanted to personally reach out about {product_info}...</p>",
                "personalization_notes": f"Fallback personalization due to error: {e}"
            }

    def predict_performance(self, subject: str, content: str, target_segment: str = "all") -> Dict:
        """Predict email performance using AI analysis"""
        
        # Get historical performance data
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT 
                c.subject,
                c.content,
                CASE WHEN c.emails_sent > 0 THEN CAST(c.opens AS FLOAT) / c.emails_sent ELSE 0 END as open_rate,
                CASE WHEN c.emails_sent > 0 THEN CAST(c.clicks AS FLOAT) / c.emails_sent ELSE 0 END as click_rate
            FROM campaigns c
            WHERE c.emails_sent > 0
        ''', conn)
        conn.close()
        
        historical_context = ""
        if not df.empty:
            avg_open_rate = df['open_rate'].mean()
            avg_click_rate = df['click_rate'].mean()
            historical_context = f"Historical averages - Open rate: {avg_open_rate:.2%}, Click rate: {avg_click_rate:.2%}"
        
        prompt = f"""
        Predict email campaign performance:
        
        SUBJECT LINE: "{subject}"
        CONTENT PREVIEW: "{content[:300]}..."
        TARGET SEGMENT: {target_segment}
        
        HISTORICAL CONTEXT: {historical_context}
        
        Analyze and predict:
        1. Open rate (0-1): Based on subject line appeal
        2. Click rate (0-1): Based on content quality and CTA
        3. Engagement score (0-1): Overall content engagement potential
        4. Risk factors: What might hurt performance
        5. Improvement suggestions
        
        Return as JSON:
        {{
            "predicted_open_rate": 0.25,
            "predicted_click_rate": 0.05,
            "engagement_score": 0.7,
            "risk_factors": ["factor1", "factor2"],
            "improvements": ["suggestion1", "suggestion2"],
            "confidence": "medium"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI email performance analyst with expertise in email marketing metrics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            return {
                "predicted_open_rate": 0.2,
                "predicted_click_rate": 0.03,
                "engagement_score": 0.5,
                "risk_factors": ["API analysis unavailable"],
                "improvements": ["Unable to generate suggestions"],
                "confidence": "low"
            }

    def optimize_send_time(self, recipient_email: str) -> Dict:
        """Find optimal send time for individual recipient"""
        
        # Get more detailed timing analysis
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT eh.send_hour, eh.opened, eh.clicked
            FROM email_history eh
            JOIN recipients r ON eh.recipient_id = r.id
            WHERE r.email = ?
        ''', (recipient_email,))
        
        timing_data = cursor.fetchall()
        conn.close()
        
        if not timing_data:
            return {
                "optimal_hour": 10,
                "confidence": "low",
                "reason": "No historical data available"
            }
        
        # Find hours with highest engagement
        hour_performance = {}
        for hour, opened, clicked in timing_data:
            if hour not in hour_performance:
                hour_performance[hour] = {"total": 0, "engaged": 0}
            hour_performance[hour]["total"] += 1
            if opened or clicked:
                hour_performance[hour]["engaged"] += 1
        
        # Calculate engagement rate by hour
        best_hour = 10
        best_rate = 0
        
        for hour, data in hour_performance.items():
            if data["total"] >= 2:  # Need at least 2 emails for reliability
                rate = data["engaged"] / data["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_hour = hour
        
        confidence = "high" if best_rate > 0.3 else "medium" if best_rate > 0.1 else "low"
        
        return {
            "optimal_hour": best_hour,
            "confidence": confidence,
            "reason": f"Best engagement rate: {best_rate:.1%} at {best_hour}:00"
        }

    def create_campaign(self, name: str, recipients: List[str], **kwargs) -> int:
        """Create AI-optimized campaign"""
        
        print(f"🤖 Creating AI-optimized campaign: {name}")
        
        # For demo, use the first recipient to generate content
        if recipients:
            sample_email = self.generate_personalized_email(
                recipient_email=recipients[0],
                campaign_purpose=kwargs.get('purpose', 'General update'),
                product_info=kwargs.get('product', 'Our service'),
                key_benefits=kwargs.get('benefits', [])
            )
            
            subject = sample_email.get('subject', 'Important Update')
            content = sample_email.get('body', 'Content generation failed')
            
            # Get AI insights
            performance = self.predict_performance(subject, content)
            insights = {
                "predicted_open_rate": performance['predicted_open_rate'],
                "predicted_click_rate": performance['predicted_click_rate'],
                "personalization_strategy": sample_email.get('personalization_notes', 'N/A')
            }
            
            # Save campaign
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO campaigns (name, subject, content, ai_insights)
                VALUES (?, ?, ?, ?)
            ''', (name, subject, content, json.dumps(insights)))
            
            campaign_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Campaign created with ID: {campaign_id}")
            print(f"📊 Predicted open rate: {performance['predicted_open_rate']:.1%}")
            print(f"📊 Predicted click rate: {performance['predicted_click_rate']:.1%}")
            
            return campaign_id
        
        return 0

    def send_campaign(self, campaign_id: int, smtp_config: Dict, test_mode: bool = True):
        """Send campaign with AI-optimized timing"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get campaign
        cursor.execute('SELECT * FROM campaigns WHERE id = ?', (campaign_id,))
        campaign = cursor.fetchone()
        
        if not campaign:
            print("❌ Campaign not found")
            return
        
        # Get recipients
        cursor.execute('SELECT * FROM recipients WHERE subscribed = 1 OR subscribed IS NULL')
        recipients = cursor.fetchall()
        
        if test_mode:
            recipients = recipients[:2]
            print(f"🧪 TEST MODE: Sending to {len(recipients)} recipients")
        
        # Setup SMTP
        try:
            context = ssl.create_default_context()
            server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
            server.starttls(context=context)
            server.login(smtp_config['email'], smtp_config['password'])
            
            sent_count = 0
            
            for recipient in recipients:
                try:
                    # Get optimal send time for this recipient
                    optimal_timing = self.optimize_send_time(recipient[1])
                    
                    # Generate personalized content
                    personalized = self.generate_personalized_email(
                        recipient_email=recipient[1],
                        campaign_purpose="Product update",
                        product_info="Our latest features"
                    )
                    
                    # Create email
                    message = MIMEMultipart()
                    message["Subject"] = personalized.get('subject', campaign[2])
                    message["From"] = smtp_config['email']
                    message["To"] = recipient[1]
                    
                    body = personalized.get('body', campaign[3])
                    message.attach(MIMEText(body, "html"))
                    
                    # Send email
                    server.send_message(message)
                    
                    # Record send
                    cursor.execute('''
                        INSERT INTO email_history (recipient_id, campaign_id, send_hour)
                        VALUES (?, ?, ?)
                    ''', (recipient[0], campaign_id, datetime.now().hour))
                    
                    sent_count += 1
                    print(f"✅ Sent to {recipient[1]} (optimal time: {optimal_timing['optimal_hour']}:00)")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"❌ Failed to send to {recipient[1]}: {e}")
                    continue
            
            server.quit()
            
            # Update campaign
            cursor.execute('''
                UPDATE campaigns SET emails_sent = ? WHERE id = ?
            ''', (sent_count, campaign_id))
            
            conn.commit()
            print(f"🎉 Campaign sent! {sent_count} emails delivered")
            
        except Exception as e:
            print(f"❌ SMTP error: {e}")
        
        conn.close()

    def get_analytics(self):
        """Get campaign analytics"""
        conn = sqlite3.connect(self.db_path)
        
        campaigns_df = pd.read_sql_query('''
            SELECT 
                c.id,
                c.name,
                c.subject,
                c.emails_sent,
                c.opens,
                c.clicks,
                c.ai_insights,
                c.created_at
            FROM campaigns c
            ORDER BY c.created_at DESC
        ''', conn)
        
        recipients_df = pd.read_sql_query('''
            SELECT 
                industry,
                COUNT(*) as count,
                AVG(engagement_score) as avg_engagement
            FROM recipients
            GROUP BY industry
        ''', conn)
        
        conn.close()
        
        return {
            "campaigns": campaigns_df,
            "recipients_by_industry": recipients_df
        }

# Demo function
def demo_ai_email_bot():
    """Demo the AI email bot with practical examples"""
    
    print("🤖 AI Email Marketing Bot - Practical Demo")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("⚠️ Skipping AI features - no API key provided")
        return
    
    # Initialize bot
    bot = AIEmailBot(api_key)
    
    # Add sample recipients
    sample_recipients = [
        {
            "email": "john@techstartup.com",
            "name": "John Smith",
            "company": "TechStartup Inc",
            "industry": "Technology"
        },
        {
            "email": "sarah@marketing.com",
            "name": "Sarah Johnson",
            "company": "Marketing Solutions",
            "industry": "Marketing"
        },
        {
            "email": "jai@marketing.com",
            "name": "Jai",
            "company": "Tech Solutions",
            "industry": "Technology"
        }
    ]
    
    # bot.add_recipients(sample_recipients)
    
    # Demo 1: Behavioral Analysis for user with no history
    print("\nDemo 1: AI Behavioral Analysis")
    print("-" * 40)
    behavior = bot.analyze_recipient_behavior("john@techstartup.com")
    print(f"Engagement Score: {behavior['engagement_score']:.2f}")
    print(f"Preferred Tone: {behavior['preferred_tone']}")
    print(f"Best Send Hour: {behavior['best_send_hour']}:00")

    # Demo 2: Behavioral Analysis for User with History
    print("\nDemo 2: AI Behavioral Analysis for User with History")
    print("-" * 40)
    behavior = bot.analyze_recipient_behavior("jai@marketing.com")
    print(f"Engagement Score: {behavior['engagement_score']:.2f}")
    print(f"Preferred Tone: {behavior['preferred_tone']}")
    print(f"Best Send Hour: {behavior['best_send_hour']}:00")
    
    # Demo 3: Personalized Content
    print("\nDemo 3: AI-Personalized Content")
    print("-" * 40)
    personalized = bot.generate_personalized_email(
        recipient_email="john@techstartup.com",
        campaign_purpose="Product launch announcement",
        product_info="AI-powered project management tool",
        key_benefits=["Save 5 hours per week", "Automate task scheduling", "Real-time team insights"]
    )
    print(f"Subject: {personalized.get('subject', 'N/A')}")
    print(f"Personalization Strategy: {personalized.get('personalization_notes', 'N/A')}")
    
    # Demo 4: Performance Prediction
    print("\nDemo 4: AI Performance Prediction")
    print("-" * 40)
    prediction = bot.predict_performance(
        subject=personalized.get('subject', 'Test Subject'),
        content=personalized.get('body', 'Test content'),
        target_segment="technology"
    )
    print(f"Predicted Open Rate: {prediction['predicted_open_rate']:.1%}")
    print(f"Predicted Click Rate: {prediction['predicted_click_rate']:.1%}")
    print(f"Confidence: {prediction['confidence']}")
    
    # Demo 5: Send Time Optimization
    print("\nDemo 5: AI Send Time Optimization")
    print("-" * 40)
    timing = bot.optimize_send_time("john@techstartup.com")
    print(f"Optimal Hour: {timing['optimal_hour']}:00")
    print(f"Confidence: {timing['confidence']}")
    print(f"Reason: {timing['reason']}")
    
    print("\nAI Demo Complete!")
    print("\nWhat makes this AI-powered vs templates:")
    print("• Learns from individual recipient behavior")
    print("• Adapts tone and content to preferences")
    print("• Predicts performance before sending")
    print("• Optimizes send times per person")
    print("• Continuously improves with data")
    
    return bot

if __name__ == "__main__":
    demo_ai_email_bot()